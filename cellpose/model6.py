"""
Copyright © 2026 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
Refactored to Dual-Parasite Swin-FPN Architecture for Multi-Scale Extraction.
"""

import os, time
from pathlib import Path
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.ndimage import gaussian_filter
import gc
import cv2
import copy 
import timm

import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .core import assign_device, run_net, run_3D

_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home().joinpath(".cellpose", "models")
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = ["swin_dual"]
MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath("gui_models.txt"))

normalize_default = {
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False
}


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


# =================================================================
# FPN PARASITE DECODERS
# =================================================================

class ParasiteDecoder(nn.Module):
    """
    Lightweight Feature Pyramid Network (FPN) upsampler.
    Ingests 3 specific hierarchical stages from the Swin backbone,
    fuses them top-down, and outputs the final 3-channel flow map.
    """
    def __init__(self, in_channels_list, final_upsample_factor, out_channels=3):
        super().__init__()
        # in_channels_list expects [C_high, C_mid, C_low] (e.g., [1024, 512, 256])
        
        self.up_high = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_high_mid = nn.Sequential(
            nn.Conv2d(in_channels_list[0] + in_channels_list[1], in_channels_list[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels_list[1]),
            nn.ReLU(inplace=True)
        )

        self.up_mid = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_mid_low = nn.Sequential(
            nn.Conv2d(in_channels_list[1] + in_channels_list[2], in_channels_list[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels_list[2]),
            nn.ReLU(inplace=True)
        )

        # Upsample back to native image resolution (H, W)
        self.final_up = nn.Upsample(scale_factor=final_upsample_factor, mode='bilinear', align_corners=False)
        
        # Output: [Flow_Y, Flow_X, Cellprob]
        self.final_conv = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)

    def forward(self, x_high, x_mid, x_low):
        # 1. Top-Down Fusion (High to Mid)
        x = self.up_high(x_high)
        x = torch.cat([x, x_mid], dim=1)
        x = self.conv_high_mid(x)

        # 2. Top-Down Fusion (Mid to Low)
        x = self.up_mid(x)
        x = torch.cat([x, x_low], dim=1)
        x = self.conv_mid_low(x)

        # 3. Final Native Projection
        x = self.final_up(x)
        return self.final_conv(x)


class DualParasiteSwinNetwork(nn.Module):
    """
    Houses the frozen Swin feature quarry and routes the stages 
    to the independent Organelle and Cell parasite decoders.
    """
    def __init__(self, nchan=3, learn_volcano=True, alpha=1.0, beta=0.1):
        super().__init__()
        self._true_dtype = torch.float32
        self.active_head = 'both'
        
        # 1. The Swin Backbone (Universal Feature Quarry)
        # Using base patch4 window7. Native feature dimensions: [128, 256, 512, 1024]
        models_logger.info(f">>> Instantiating Swin-Base Backbone ({nchan} channels)...")
        self.backbone = timm.create_model('swin_base_patch4_window7_224', in_chans=nchan, features_only=True, pretrained=True)
        
        # 2. The Cell Parasite (Hooks into deep semantics: Stages 4, 3, 2)
        # Stage 2 native resolution is 1/8, so final upsample is 8x
        self.cell_decoder = ParasiteDecoder(in_channels_list=[1024, 512, 256], final_upsample_factor=8)
        
        # 3. The Organelle Parasite (Hooks into high-res details: Stages 3, 2, 1)
        # Stage 1 native resolution is 1/4, so final upsample is 4x
        self.org_decoder = ParasiteDecoder(in_channels_list=[512, 256, 128], final_upsample_factor=4)

        # 4. Volcano Merger
        self.learn_volcano = learn_volcano
        if self.learn_volcano:
            self.alpha = nn.Parameter(torch.tensor([float(alpha)], dtype=self._true_dtype))
            self.beta = nn.Parameter(torch.tensor([float(beta)], dtype=self._true_dtype))
            models_logger.info(f">>> [VOLCANO MERGER] Learnable Mode Active (alpha={alpha}, beta={beta})")
        else:
            self.register_buffer('alpha', torch.tensor([float(alpha)], dtype=self._true_dtype))
            self.register_buffer('beta', torch.tensor([float(beta)], dtype=self._true_dtype))

    @property
    def device(self):
        return next(self.parameters()).device

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def apply_volcano_merger(self, out_c):
        import torchvision.transforms.functional as TF
        
        cell_flows = out_c[:, :2, :, :]
        cell_logits = out_c[:, 2:, :, :]
        
        prob = torch.sigmoid(cell_logits)
        prob_dome = TF.gaussian_blur(prob, kernel_size=15, sigma=[3.0, 3.0])
        
        grad_y = torch.zeros_like(prob_dome)
        grad_x = torch.zeros_like(prob_dome)
        
        grad_y[:, :, 1:-1, :] = (prob_dome[:, :, 2:, :] - prob_dome[:, :, :-2, :]) / 2.0
        grad_x[:, :, :, 1:-1] = (prob_dome[:, :, :, 2:] - prob_dome[:, :, :, :-2]) / 2.0
        
        dome_flows = torch.cat([grad_y, grad_x], dim=1)
        merged_flows = (torch.abs(self.alpha) * cell_flows) + (torch.abs(self.beta) * dome_flows)
        
        norm = torch.norm(merged_flows, p=2, dim=1, keepdim=True)
        merged_flows = merged_flows / (norm + 1e-8)
        merged_flows = merged_flows * 5.0 * prob
        
        return torch.cat([merged_flows, cell_logits], dim=1)

    def forward(self, x):
        x = x.to(self._true_dtype)
        
        # --- PASS 1: EVALUATION FEEDBACK (If requested) ---
        if not self.training and self.active_head in ['cells', 'both']:
            with torch.no_grad():
                feats_pass1 = self.backbone(x)
                # Cell hook mapping: High=S4(idx 3), Mid=S3(idx 2), Low=S2(idx 1)
                out_c_pass1 = self.cell_decoder(feats_pass1[3], feats_pass1[2], feats_pass1[1])
                prob_map = torch.sigmoid(out_c_pass1[:, 2:3, :, :])
                x_target = x * prob_map
        else:
            x_target = x
            
        # --- PASS 2: MAIN FEATURE EXTRACTION ---
        # The backbone is completely frozen, so we don't track its gradients to save massive VRAM
        with torch.no_grad():
            features = self.backbone(x_target)
            
        s1, s2, s3, s4 = features[0], features[1], features[2], features[3]

        out_c, out_o = None, None
        
        # Fake "styles" for Cellpose compatibility by pooling the deepest feature map used
        style_c = torch.mean(s4, dim=(2, 3))
        style_o = torch.mean(s3, dim=(2, 3))

        # --- PARASITE ROUTING ---
        if self.active_head in ['cells', 'both']:
            out_c = self.cell_decoder(s4, s3, s2)
            out_c = self.apply_volcano_merger(out_c)

        if self.active_head in ['organelles', 'both']:
            out_o = self.org_decoder(s3, s2, s1)
            out_o = self.apply_volcano_merger(out_o)

        if self.active_head == 'cells':
            return out_c, style_c
        elif self.active_head == 'organelles':
            return out_o, style_o
        else:
            return (out_c, out_o), (style_c, style_o)


# =================================================================
# CELLPOSE MODEL WRAPPER
# =================================================================

class CellposeModel():
    def __init__(self, gpu=False, pretrained_model=None, custom_weights=None, model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True, manual=True, 
                 freeze_backbone=True, learn_volcano=True, alpha=1.0, beta=0.1):

        if diam_mean is not None:
            models_logger.warning("diam_mean argument is deprecated in this architecture.")
        
        self.nchan = nchan if nchan is not None else 3
        
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu

        self.net = DualParasiteSwinNetwork(
            nchan=self.nchan, 
            learn_volcano=learn_volcano, 
            alpha=alpha, 
            beta=beta
        ).to(self.device)

        if use_bfloat16 and self.gpu:
            self.net.to(torch.bfloat16)
            self.net._true_dtype = torch.bfloat16

        # --- WEIGHT LOADING ---
        if custom_weights is not None and os.path.exists(custom_weights):
            models_logger.info(f">>>> Loading CUSTOM Dual-Parasite weights: {custom_weights}")
            self.net.load_model(custom_weights, device=self.device)
        elif pretrained_model is not None and os.path.exists(pretrained_model):
            models_logger.info(f">>>> Loading Pretrained weights: {pretrained_model}")
            self.net.load_model(pretrained_model, device=self.device)

        # --- FREEZE LOGIC ---
        self.set_freeze_backbone(freeze_backbone)
        

    def set_freeze_backbone(self, freeze=True):
        self.freeze_backbone = freeze
        if freeze:
            models_logger.info("\n>>> [MODELS] BACKBONE FROZEN: Universal Feature Quarry active. Only FPN Parasites will train.")
        else:
            models_logger.info("\n>>> [MODELS] UNFREEZING BACKBONE: Warning - Swin Backbone is training End-to-End.")
            
        # Lock or unlock backbone
        for param in self.net.backbone.parameters():
            param.requires_grad = not freeze
            
        # Parasites ALWAYS train
        for param in self.net.cell_decoder.parameters():
            param.requires_grad = True
        for param in self.net.org_decoder.parameters():
            param.requires_grad = True

        # Volcano parameters
        for name, param in self.net.named_parameters():
            if name in ['alpha', 'beta']:
                param.requires_grad = getattr(self.net, 'learn_volcano', False)

        
    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, 
             min_size=15, max_size_fraction=0.4, niter=None, 
             augment=False, tile_overlap=0.1, bsize=256, 
             compute_masks=True, progress=None,
             active_head='cells', visualize=False, ground_truth=None):
             
        if isinstance(x, list) or x.squeeze().ndim == 5:
            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out, mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i], 
                    batch_size=batch_size,
                    channel_axis=channel_axis, 
                    z_axis=z_axis,
                    normalize=normalize, 
                    invert=invert,
                    diameter=diameter[i] if isinstance(diameter, (list, np.ndarray)) else diameter, 
                    do_3D=do_3D,
                    anisotropy=anisotropy, 
                    augment=augment, 
                    tile_overlap=tile_overlap, 
                    bsize=bsize, 
                    resample=resample,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, 
                    compute_masks=compute_masks,
                    min_size=min_size, 
                    max_size_fraction=max_size_fraction, 
                    stitch_threshold=stitch_threshold, 
                    flow3D_smooth=flow3D_smooth,
                    progress=progress, 
                    niter=niter,
                    active_head=active_head,
                    visualize=visualize,
                    ground_truth=[ground_truth[i]] if ground_truth is not None else None)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        ############# actual eval code ############
        raw_x = np.copy(x)

        if x.ndim == 3 and x.shape[0] in [2, 3, 6]: 
            x = x.transpose(1, 2, 0)
        if x.ndim == 4 and x.shape[1] in [2, 3, 6]: 
            x = x.transpose(0, 2, 3, 1)
        
        if x.ndim == 3 and x.shape[-1] == 3 and np.max(x[..., 2]) == 0 and np.min(x[..., 2]) == 0: 
            x = x[..., :2]
        if x.ndim == 4 and x.shape[-1] == 3 and np.max(x[..., 2]) == 0 and np.min(x[..., 2]) == 0: 
            x = x[..., :2]

        if x.ndim == 3 and x.shape[-1] == 2:
            if self.nchan == 6:
                x = np.concatenate([np.repeat(x[..., 0:1], 3, axis=-1), np.repeat(x[..., 1:2], 3, axis=-1)], axis=-1)
            else:
                x = np.concatenate([x[..., 0:1], x[..., 1:2], np.zeros_like(x[..., 0:1])], axis=-1)
        elif x.ndim == 4 and x.shape[-1] == 2:
            if self.nchan == 6:
                x = np.concatenate([np.repeat(x[..., 0:1], 3, axis=-1), np.repeat(x[..., 1:2], 3, axis=-1)], axis=-1)
            else:
                x = np.concatenate([x[..., 0:1], x[..., 1:2], np.zeros_like(x[..., 0:1])], axis=-1)

        if x.ndim < 4:
            x = x[np.newaxis, ...]
        nimg = x.shape[0]
        
        # ---> DECOUPLED SCALING FACTOR CALCULATIONS <---
        image_scaling = {'cells': 1.0, 'organelles': 1.0}
        if diameter is not None:
            if isinstance(diameter, (list, tuple, np.ndarray)) and len(diameter) >= 2:
                if diameter[0] > 0: image_scaling['cells'] = 30. / diameter[0]
                if diameter[1] > 0: image_scaling['organelles'] = 30. / diameter[1]
            elif isinstance(diameter, dict):
                if diameter.get('cells', 0) > 0: image_scaling['cells'] = 30. / diameter['cells']
                if diameter.get('organelles', 0) > 0: image_scaling['organelles'] = 30. / diameter['organelles']
            elif isinstance(diameter, (int, float)) and diameter > 0:
                image_scaling['cells'] = 30. / diameter
                image_scaling['organelles'] = 30. / diameter

        normalize_params = normalize_default
        if isinstance(normalize, dict):
            normalize_params = {**normalize_params, **normalize}
        elif not isinstance(normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params["normalize"] = normalize
            normalize_params["invert"] = invert

        do_normalization = True if normalize_params["normalize"] else False
        if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
            normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
            x = transforms.normalize_img(x, **normalize_params)
            do_normalization = False 
        else:
            if normalize_params["norm3D"] and nimg > 1 and do_normalization:
                normalize_params["norm3D"] = False
        if do_normalization:
            x = transforms.normalize_img(x, **normalize_params)

        if hasattr(self.net, 'eval'):
            self.net.eval()

        network_outputs = self._run_net(
            x,
            resample=resample,
            rescale=image_scaling,
            augment=augment, 
            batch_size=batch_size, 
            tile_overlap=tile_overlap, 
            bsize=bsize,
            do_3D=do_3D, 
            anisotropy=anisotropy,
            active_head=active_head) 

        all_masks = []
        all_flows = []
        all_styles = []

        heads_to_process = ['cells', 'organelles'] if active_head == 'both' else [active_head]

        for head in heads_to_process:
            dP, cellprob, styles = network_outputs[head]

            if do_3D and flow3D_smooth:
                if isinstance(flow3D_smooth, (int, float)):
                    flow3D_smooth = [flow3D_smooth]*3 
                if isinstance(flow3D_smooth, list) and len(flow3D_smooth) == 1:
                    flow3D_smooth = flow3D_smooth*3
                if len(flow3D_smooth) == 3 and any(v > 0 for v in flow3D_smooth):
                    dP = gaussian_filter(dP, [0, *flow3D_smooth])
                torch.cuda.empty_cache()
                gc.collect()

            if compute_masks:
                current_scale = image_scaling.get(head, 1.0)
                niter_scale = 1 if current_scale is None else current_scale
                niter_val = int(200/niter_scale) if niter is None or niter == 0 else niter
                masks = self._compute_masks(x.shape, dP, cellprob, 
                                            flow_threshold=flow_threshold,
                                            cellprob_threshold=cellprob_threshold, 
                                            min_size=min_size,
                                            max_size_fraction=max_size_fraction, 
                                            niter=niter_val,
                                            stitch_threshold=stitch_threshold, 
                                            do_3D=do_3D)
            else:
                masks = np.zeros(0) 
            
            masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()
            all_masks.append(masks)
            all_flows.append([plot.dx_to_circ(dP), dP, cellprob])
            all_styles.append(styles)

        # --- VISUALIZATION OVERLAY ---
        if visualize:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches

                img_display = raw_x.squeeze()
                if img_display.ndim > 2 and img_display.shape[0] in [2, 3, 4, 6]:
                    img_display = img_display.transpose(1, 2, 0)
                    
                if img_display.max() > 1.0:
                    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

                n_rows = 2 if active_head == 'both' else 1
                fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6 * n_rows))
                
                if n_rows == 1:
                    axes = [axes] 

                for row, head in enumerate(heads_to_process):
                    pred_mask = all_masks[row]
                    chan_idx = 0 if head == 'cells' else (3 if self.nchan == 6 else 1)
                    if img_display.ndim == 3 and img_display.shape[-1] > chan_idx:
                        img_show = img_display[..., chan_idx]
                    else:
                        img_show = img_display[..., 0]
                    
                    axes[row][0].imshow(img_show, cmap='gray')
                    title_gt = f"Original Input ({head})"
                    
                    if ground_truth is not None:
                        if isinstance(ground_truth, list) and len(ground_truth) > row:
                            true_mask = ground_truth[row].squeeze()
                        elif isinstance(ground_truth, np.ndarray) and ground_truth.shape[0] == n_rows:
                            true_mask = ground_truth[row].squeeze()
                        else:
                            true_mask = ground_truth[0].squeeze() if isinstance(ground_truth, list) else ground_truth.squeeze()
                            
                        if np.any(true_mask > 0):
                            axes[row][0].contour(true_mask, levels=np.unique(true_mask), colors='lime', linewidths=0.5, alpha=0.8)
                        title_gt = f"Actual GT Overlay ({head})"
                    
                    axes[row][0].set_title(title_gt, fontweight='bold')
                    axes[row][0].axis('off')

                    axes[row][1].imshow(img_show, cmap='gray')
                    if np.any(pred_mask > 0):
                        axes[row][1].contour(pred_mask, levels=np.unique(pred_mask), colors='red', linewidths=0.5, alpha=0.8)
                    axes[row][1].set_title(f"Predicted Overlay ({head})", fontweight='bold')
                    axes[row][1].axis('off')
                    
                    if ground_truth is not None and 'true_mask' in locals():
                        true_bin = true_mask > 0
                        pred_bin = pred_mask > 0
                        intersection = np.logical_and(true_bin, pred_bin).sum()
                        union = np.logical_or(true_bin, pred_bin).sum()
                        iou = intersection / union if union > 0 else 0.0
                        
                        precision = intersection / pred_bin.sum() if pred_bin.sum() > 0 else 0.0
                        recall = intersection / true_bin.sum() if true_bin.sum() > 0 else 0.0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        legend_patch = mpatches.Patch(color='red', label=f'Pixel F1: {f1:.4f} | IoU: {iou:.4f}')
                        axes[row][1].legend(handles=[legend_patch], loc='lower right', framealpha=0.9)

                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                models_logger.warning(f"Visualization failed: {e}")

        if active_head == 'both':
            return all_masks, all_flows, all_styles
        else:
            return all_masks[0], all_flows[0], all_styles[0]
    

    def _run_net(self, x, 
                 rescale=1.0,
                 resample=True,
                 augment=False, 
                 batch_size=8, 
                 tile_overlap=0.1,
                 bsize=256, 
                 anisotropy=1.0, 
                 do_3D=False,
                 active_head='cells'):
        """ run network on image x """
        tic = time.time()
        shape = x.shape
        
        outputs = {}
        heads_to_run = ['cells', 'organelles'] if active_head == 'both' else [active_head]

        for head in heads_to_run:
            if hasattr(self.net, 'active_head'):
                self.net.active_head = head

            current_rescale = rescale.get(head, 1.0) if isinstance(rescale, dict) else rescale

            if do_3D:
                Lz, Ly, Lx = shape[:-1]
                if current_rescale != 1.0 or (anisotropy is not None and anisotropy != 1.0):
                    anisotropy = 1.0 if anisotropy is None else anisotropy
                    if current_rescale != 1.0:
                        x_in = transforms.resize_image(x, Ly=int(Ly*current_rescale), Lx=int(Lx*current_rescale))
                    else:
                        x_in = x
                    x_in = transforms.resize_image(x_in.transpose(1,0,2,3),
                                                   Ly=int(Lz*anisotropy*current_rescale), 
                                                   Lx=int(Lx*current_rescale)).transpose(1,0,2,3)
                else:
                    x_in = x
                yf, styles = run_3D(self.net, x_in,
                                    batch_size=batch_size, augment=augment,  
                                    tile_overlap=tile_overlap, 
                                    bsize=bsize)
            else:
                yf, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                                    batch_size=batch_size,  
                                    tile_overlap=tile_overlap, 
                                    rsz=current_rescale if current_rescale != 1.0 else None)

            if resample:
                if do_3D:
                    if current_rescale != 1.0 or Lz != yf.shape[0]:
                        if current_rescale != 1.0:
                            yf = transforms.resize_image(yf, Ly=Ly, Lx=Lx)
                        if Lz != yf.shape[0]:
                            yf = transforms.resize_image(yf.transpose(1, 0, 2, 3), Ly=Lz, Lx=Lx).transpose(1, 0, 2, 3)
                else:
                    if current_rescale != 1.0:
                        yf = transforms.resize_image(yf, shape[1], shape[2])
            
            if do_3D:
                cellprob = yf[..., -1]
                dP = yf[..., :-1].transpose((3, 0, 1, 2))
            else:
                cellprob = yf[..., -1]
                dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
                
            outputs[head] = (dP, cellprob, styles.squeeze() if isinstance(styles, np.ndarray) else styles)

        if hasattr(self.net, 'active_head'):
            self.net.active_head = active_head

        return outputs
    
    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
        """ compute masks from flows and cell probability """
        changed_device_from = None
        if self.device.type == "mps" and do_3D:
            self.device = torch.device("cpu")
            changed_device_from = "mps"
        Lz, Ly, Lx = shape[:3]
        tic = time.time()
        if do_3D:
            masks = dynamics.resize_and_compute_masks(
                dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold, do_3D=do_3D,
                min_size=min_size, max_size_fraction=max_size_fraction, 
                resize=shape[:3] if (np.array(dP.shape[-3:])!=np.array(shape[:3])).sum() else None,
                device=self.device)
        else:
            nimg = shape[0]
            Ly0, Lx0 = cellprob[0].shape 
            resize = None if Ly0==Ly and Lx0==Lx else [Ly, Lx]
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out, mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                min_size0 = min_size if stitch_threshold == 0 or nimg == 1 else -1
                outputs = dynamics.resize_and_compute_masks(
                    dP[:, i], cellprob[i],
                    niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, resize=resize,
                    min_size=min_size0, max_size_fraction=max_size_fraction,
                    device=self.device)
                if i==0 and nimg > 1:
                    masks = np.zeros((nimg, shape[1], shape[2]), outputs.dtype)
                if nimg > 1:
                    masks[i] = outputs
                else:
                    masks = outputs

            if stitch_threshold > 0 and nimg > 1:
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                masks = utils.fill_holes_and_remove_small_masks(masks, min_size=min_size)

        if changed_device_from is not None:
            self.device = torch.device(changed_device_from)
        return masks
