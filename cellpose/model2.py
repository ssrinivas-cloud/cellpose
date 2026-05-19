"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import os, time
from pathlib import Path
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import gc
import cv2
import copy 

import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .vit_sam import Transformer
from .core import assign_device, run_net, run_3D

_CPSAM_MODEL_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home().joinpath(".cellpose", "models")
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = ["cpsam"]

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


def model_path(model_type, model_index=0):
    return cache_CPSAM_model_path()


def cache_CPSAM_model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = os.fspath(MODEL_DIR.joinpath('cpsam'))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(_CPSAM_MODEL_URL, cached_file))
        utils.download_url_to_file(_CPSAM_MODEL_URL, cached_file, progress=True)
    return cached_file


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


class DualPathTransformer(nn.Module):
    """
    Wraps the ViT backbone to safely split the architecture at the Neck.
    Handles the PixelShuffle reshape natively.
    """
    def __init__(self, base_net, randomize_org=False):
        super().__init__()
        self.base_net = base_net
        
        # Capture the true backbone dtype to prevent PyTorch mixed-precision bias crashes
        self._true_dtype = next(base_net.parameters()).dtype
        
        # Forward Cellpose attributes to the wrapper
        self.diam_mean = getattr(base_net, 'diam_mean', nn.Parameter(torch.tensor([30.0], dtype=self._true_dtype)))
        self.diam_labels = getattr(base_net, 'diam_labels', nn.Parameter(torch.tensor([30.0], dtype=self._true_dtype)))
        
        # Deep clone the Neck and Head for Cells (Keeps Pretrained Weights)
        self.cell_neck = copy.deepcopy(base_net.encoder.neck)
        self.cell_out = copy.deepcopy(base_net.out)
        
        # Deep clone the Neck and Head for Organelles 
        self.org_neck = copy.deepcopy(base_net.encoder.neck)
        self.org_out = copy.deepcopy(base_net.out)
        
        # ---> OPTIONAL SYMMETRY BREAKING (Ablation toggle) <---
        if randomize_org:
            models_logger.info(">>> [ABLATION] Randomizing Organelle Head Weights (Kaiming Normal)...")
            for m in self.org_neck.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
            for m in self.org_out.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
        
        # Disconnect original neck and out to prevent double-processing
        self.base_net.encoder.neck = nn.Identity()
        self.base_net.out = nn.Identity()
        
        self.active_head = 'both'

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return self._true_dtype

    # ---> NEW FIX: Add a setter so train.py can actually change the dtype!
    @dtype.setter
    def dtype(self, new_dtype):
        self._true_dtype = new_dtype
        self.to(new_dtype)

    def load_model(self, path, device):
        """ Safe hook for train.py to load the weights """
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def save_model(self, path):
        """ Safe hook for train.py to save the weights """
        torch.save(self.state_dict(), path)

    def pixel_shuffle(self, x):
        """ Recreates Cellpose's hidden 8x upsampling reshape (192 -> 3 channels) """
        B, C, H, W = x.shape
        out_c = C // 64 
        x = x.view(B, out_c, 8, 8, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(B, out_c, H * 8, W * 8)

    def forward(self, x):
        # Safety cast to prevent mixed precision errors
        x = x.to(self._true_dtype)
        
        # 1. Get raw ViT features [B, 1024, 32, 32]
        feat = self.base_net.encoder(x) 
        
        if self.active_head == 'cells':
            feat_c = self.cell_neck(feat)
            out_c = self.pixel_shuffle(self.cell_out(feat_c))
            style = torch.mean(feat_c, dim=(2, 3))
            return out_c, style
            
        elif self.active_head == 'organelles':
            feat_o = self.org_neck(feat)
            out_o = self.pixel_shuffle(self.org_out(feat_o))
            style = torch.mean(feat_o, dim=(2, 3))
            return out_o, style
            
        else:
            # === RESTORED: Both mode for Training Loop ===
            # Cell Path
            feat_c = self.cell_neck(feat)
            out_c = self.pixel_shuffle(self.cell_out(feat_c))
            style_c = torch.mean(feat_c, dim=(2, 3))
            
            # Organelle Path
            feat_o = self.org_neck(feat)
            out_o = self.pixel_shuffle(self.org_out(feat_o))
            style_o = torch.mean(feat_o, dim=(2, 3))
            
            return (out_c, out_o), (style_c, style_o)


class CellposeModel():
    """
    Class representing a Cellpose model.

    Attributes:
        diam_mean (float): Mean "diameter" value for the model.
        builtin (bool): Whether the model is a built-in model or not.
        device (torch device): Device used for model running / training.
        nclasses (int): Number of classes in the model.
        nbase (list): List of base values for the model.
        net (CPnet): Cellpose network.
        pretrained_model (str): Path to pretrained cellpose model.
        pretrained_model_ortho (str): Path or model_name for pretrained cellpose model for ortho views in 3D.
        backbone (str): Type of network ("default" is the standard res-unet, "transformer" for the segformer).
        freeze_backbone (bool): Whether the ViT backbone is frozen for head-only fine-tuning.
    """

    def __init__(self, gpu=False, pretrained_model="cpsam", custom_weights=None, model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True, manual=True, 
                 freeze_backbone=False, random=False):

        if diam_mean is not None:
            models_logger.warning("diam_mean argument are not used in v4.0.1+. Ignoring this argument...")
        if model_type is not None:
            models_logger.warning("model_type argument is not used in v4.0.1+. Ignoring this argument...")
        
        # Default to 3 channels (RGB mode), train.py will pass 6 if two_tail=True
        if nchan is None:
            nchan = 3
            
        self.nchan = nchan

        ### assign model device
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu

        if pretrained_model is None and custom_weights is None:
            raise ValueError("Must specify a pretrained model, training from scratch is not implemented")
        
        ### check for pretrained model
        if pretrained_model and not os.path.exists(pretrained_model):
            model_strings = get_user_models()
            all_models = MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if pretrained_model in all_models:
                pretrained_model = os.path.join(MODEL_DIR, pretrained_model)
            else:
                pretrained_model = os.path.join(MODEL_DIR, "cpsam")
                models_logger.warning(f"pretrained model {pretrained_model} not found, using default model")

        self.pretrained_model = pretrained_model
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        
        # 1. Initialize Base Network (3 channels)
        base_net = Transformer(dtype=dtype).to(self.device)

        # 2. Load Pretrained CPSAM backbone FIRST so we can clone its 3-channel weights
        if not (custom_weights is not None and os.path.exists(custom_weights)):
            if os.path.exists(self.pretrained_model):
                models_logger.info(f">>>> loading base model {self.pretrained_model}")
                base_net.load_model(self.pretrained_model, device=self.device)
            else:
                if os.path.split(self.pretrained_model)[-1] != 'cpsam':
                    raise FileNotFoundError('model file not recognized')
                cache_CPSAM_model_path()
                base_net.load_model(self.pretrained_model, device=self.device)

        # =================================================================
        # 3. DYNAMIC INPUT CHANNEL MODIFICATION (6-channel switch)
        # =================================================================
        if self.nchan != 3:
            models_logger.info(f"Modifying input patch_embed from 3 to {self.nchan} channels...")
            patch_embed = base_net.encoder.patch_embed
            for name, layer in patch_embed.named_children():
                if isinstance(layer, nn.Conv2d):
                    new_conv = nn.Conv2d(
                        in_channels=self.nchan,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=(layer.bias is not None)
                    )
                    with torch.no_grad():
                        # Divide weights by 2 so adding them together doesn't blow up gradients
                        half_weight = layer.weight.clone() / 2.0
                        
                        # Apply to channels 0,1,2 (Cells) and 3,4,5 (Organelles)
                        new_conv.weight[:, :3, :, :] = half_weight
                        new_conv.weight[:, 3:, :, :] = half_weight
                        
                        if layer.bias is not None:
                            new_conv.bias.copy_(layer.bias)
                            
                    setattr(patch_embed, name, new_conv)
                    getattr(patch_embed, name).to(dtype=dtype, device=self.device)
                    models_logger.info(f"Input {self.nchan}-channel modification complete!")
                    break

        # 4. Wrap in DualPath Architecture
        if not manual or custom_weights is not None:
            models_logger.info("Injecting DualPathTransformer (Branching directly from SAM Neck)...")
            self.net = DualPathTransformer(base_net, randomize_org=random).to(self.device)
        else:
            self.net = base_net

        self.freeze_backbone = freeze_backbone

        # 5. If resuming training from Custom Weights, load them NOW
        if custom_weights is not None and os.path.exists(custom_weights):
            models_logger.info(f">>>> loading CUSTOM post-architectural weights {custom_weights}")
            self.net.load_model(custom_weights, device=self.device)

        # --- APPLY FREEZE SETTING ---
        self.set_freeze_backbone(self.freeze_backbone)
        

    def set_freeze_backbone(self, freeze=True):
        """
        Dynamically freezes or unfreezes the ViT backbone.
        The deep dual-decoder paths will ALWAYS remain unfrozen.
        """
        self.freeze_backbone = freeze
        if freeze:
            models_logger.info("\n>>> [MODELS] FREEZING BACKBONE: Only the Dual Decoder Paths will be trained.")
        else:
            models_logger.info("\n>>> [MODELS] UNFREEZING BACKBONE: The entire network will be trained (End-to-End).")
            
        for name, param in self.net.named_parameters():
            if 'cell_neck' in name or 'cell_out' in name or 'org_neck' in name or 'org_out' in name:
                param.requires_grad = True  # Deep Upsampling Paths ALWAYS train
            else:
                param.requires_grad = not freeze # ViT Backbone toggles

        
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
                    diameter=diameter[i] if isinstance(diameter, list) or isinstance(diameter, np.ndarray) else diameter, 
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

        # ---> THE DYNAMIC CHANNEL EVALUATION BYPASS <---
        # 1. Force channels to the LAST axis (H, W, C) for core.run_net
        if x.ndim == 3 and x.shape[0] in [2, 3, 6]: 
            x = x.transpose(1, 2, 0)
        if x.ndim == 4 and x.shape[1] in [2, 3, 6]: 
            x = x.transpose(0, 2, 3, 1)
        
        # 2. Remove zero-padded 3rd channel if present
        if x.ndim == 3 and x.shape[-1] == 3 and np.max(x[..., 2]) == 0 and np.min(x[..., 2]) == 0: 
            x = x[..., :2]
        if x.ndim == 4 and x.shape[-1] == 3 and np.max(x[..., 2]) == 0 and np.min(x[..., 2]) == 0: 
            x = x[..., :2]

        # 3. Dynamic Padding based on self.nchan (two_tail logic)
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
        
        image_scaling = 1.0
        if diameter is not None and diameter > 0:
            image_scaling = 30. / diameter

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
                niter_scale = 1 if image_scaling is None else image_scaling
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
                
                # Safe transposition for 3 or 6 channel arrays
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
                    
                    # Extract correct channel based on dynamic nchan
                    chan_idx = 0 if head == 'cells' else (3 if self.nchan == 6 else 1)
                    if img_display.ndim == 3 and img_display.shape[-1] > chan_idx:
                        img_show = img_display[..., chan_idx]
                    else:
                        img_show = img_display[..., 0] # Fallback
                    
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

        # ---> THE MAGIC FIX: Loop over the heads so core.run_net is natively executed TWICE <---
        for head in heads_to_run:
            if hasattr(self.net, 'active_head'):
                self.net.active_head = head

            if do_3D:
                Lz, Ly, Lx = shape[:-1]
                if rescale != 1.0 or (anisotropy is not None and anisotropy != 1.0):
                    anisotropy = 1.0 if anisotropy is None else anisotropy
                    if rescale != 1.0:
                        x_in = transforms.resize_image(x, Ly=int(Ly*rescale), Lx=int(Lx*rescale))
                    else:
                        x_in = x
                    x_in = transforms.resize_image(x_in.transpose(1,0,2,3),
                                            Ly=int(Lz*anisotropy*rescale), 
                                            Lx=int(Lx*rescale)).transpose(1,0,2,3)
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
                                    rsz=rescale if rescale !=1.0 else None)

            if resample:
                if do_3D:
                    if rescale != 1.0 or Lz != yf.shape[0]:
                        if rescale != 1.0:
                            yf = transforms.resize_image(yf, Ly=Ly, Lx=Lx)
                        if Lz != yf.shape[0]:
                            yf = transforms.resize_image(yf.transpose(1, 0, 2, 3), Ly=Lz, Lx=Lx).transpose(1, 0, 2, 3)
                else:
                    if rescale != 1.0:
                        yf = transforms.resize_image(yf, shape[1], shape[2])
            
            if do_3D:
                cellprob = yf[..., -1]
                dP = yf[..., :-1].transpose((3, 0, 1, 2))
            else:
                cellprob = yf[..., -1]
                dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
                
            outputs[head] = (dP, cellprob, styles.squeeze() if isinstance(styles, np.ndarray) else styles)

        # Re-set back to original state to prevent accidental state corruption
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
