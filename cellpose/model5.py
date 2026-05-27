"""
Copyright © 2026 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
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


# =================================================================
# HYBRID ORGANELLE MODULES (Cross-Layer Attn + ASPP + Transformer)
# =================================================================

class CrossLayerAttentionConcat(nn.Module):
    """
    Treats the L different layers as a sequence of L tokens for each spatial pixel.
    Applies Self-Attention across the layers to let them communicate,
    then concatenates the attended features and fuses them.
    """
    def __init__(self, embed_dim=1024, num_layers=7, num_heads=8):
        super().__init__()
        self.num_layers = num_layers
        # Self-Attention across the layers
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Attention-based concat reduction
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * num_layers, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: list of L tensors of shape (B, C, H, W)
        B, C, H, W = features[0].shape
        
        # Stack to (B, L, C, H, W)
        stacked = torch.stack(features, dim=1)
        
        # Rearrange to treat each pixel as a batch element with sequence length L
        # (B, L, C, H, W) -> (B, H, W, L, C) -> (B*H*W, L, C)
        x = stacked.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.num_layers, C)
        
        # Cross-Layer Self-Attention
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out # Residual connection
        
        # Reshape back to spatial multi-layer: (B*H*W, L, C) -> (B, H, W, L, C) -> (B, L, C, H, W)
        x = x.view(B, H, W, self.num_layers, C).permute(0, 3, 4, 1, 2)
        
        # Attention-based CONCAT: (B, L*C, H, W)
        x_concat = x.reshape(B, self.num_layers * C, H, W)
        
        # Fuse back down to (B, C, H, W)
        return self.fusion_conv(x_concat)


class TransformerBlock(nn.Module):
    """ Standard Multi-Head Self Attention + MLP Block """
    def __init__(self, dim=1024, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DilatedLocalExtractor(nn.Module):
    """ 
    ASPP Block using Dilated Convolutions to process local high-frequency details.
    """
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.mixer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        merged = torch.cat([b1, b2, b3, b4], dim=1)
        mixed = self.mixer(merged)
        return x + self.gamma * mixed


class HybridOrganelleTransformer(nn.Module):
    """ Alternates strictly between Transformer Blocks and ASPP blocks for exactly 2 processing layers """
    def __init__(self, embed_dim=1024, depth=2, num_heads=16, mlp_ratio=4.0, use_aspp=False):
        super().__init__()
        self.depth = depth
        self.use_aspp = use_aspp
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        if self.use_aspp:
            self.aspp_blocks = nn.ModuleList([
                DilatedLocalExtractor(embed_dim=embed_dim)
                for _ in range(depth)
            ])
            
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        for i in range(self.depth):
            # 1. Transformer Sequence Mapping
            x_seq = x.view(B, C, -1).transpose(1, 2)
            x_seq = self.transformer_blocks[i](x_seq)
            
            # 2. Re-map back to Spatial Grid
            x = x_seq.transpose(1, 2).view(B, C, H, W).contiguous()
            
            # 3. Micro-scale ASPP Injection
            if self.use_aspp:
                x = self.aspp_blocks[i](x)
                
        # Final Layer Projections
        x_seq = x.view(B, C, -1).transpose(1, 2)
        x_seq = self.norm(x_seq)
        x = x_seq.transpose(1, 2).view(B, C, H, W).contiguous()
        
        return x


class IntermediateDecoder(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


# =================================================================
# DUAL PATH ARCHITECTURE
# =================================================================

class DualPathTransformer(nn.Module):
    """
    Wraps the ViT backbone to securely branch outputs into three heads:
    1. cell_head (trainable)
    2. cell_predict (frozen eval anchor)
    3. organelle (trainable early-hook fusion via CrossLayerAttentionConcat -> 2x Transformer+ASPP layers)
    """
    def __init__(self, base_net, randomize_org=False, learn_volvolcano=True, alpha=1.0, beta=0.1, use_aspp=False):
        super().__init__()
        self.base_net = base_net
        self._true_dtype = next(base_net.parameters()).dtype
        
        self.diam_mean = getattr(base_net, 'diam_mean', nn.Parameter(torch.tensor([30.0], dtype=self._true_dtype)))
        self.diam_labels = getattr(base_net, 'diam_labels', nn.Parameter(torch.tensor([30.0], dtype=self._true_dtype)))
        
        # ---------------------------------------------------
        # 0. SETUP HOOKS FOR 7 TARGET LAYERS (CRITICAL FIX HERE)
        # ---------------------------------------------------
        # Shifted final target layer from 24 to 23 to respect zero-indexing ceiling of ViT-Large
        self.target_layers = [0, 4, 8, 12, 16, 20, 23]
        self.intermediate_features = {}
        
        for layer_idx in self.target_layers:
            self.base_net.encoder.blocks[layer_idx].register_forward_hook(
                self._get_hook(layer_idx)
            )
            
        # ===================================================
        # HEAD 1: CELL HEAD (TRAINING)
        # ===================================================
        self.cell_neck = copy.deepcopy(base_net.encoder.neck)
        self.cell_out = copy.deepcopy(base_net.out)
        
        # ===================================================
        # HEAD 2: CELL PREDICT HEAD (EVALUATION / FROZEN)
        # ===================================================
        self.cell_predict_neck = copy.deepcopy(base_net.encoder.neck)
        self.cell_predict_out = copy.deepcopy(base_net.out)
        
        # ===================================================
        # HEAD 3: ORGANELLE HEAD
        # ===================================================
        self.layer_attention_concat = CrossLayerAttentionConcat(embed_dim=1024, num_layers=len(self.target_layers)).to(self._true_dtype)
        self.organelle_transformer = HybridOrganelleTransformer(embed_dim=1024, depth=2, use_aspp=use_aspp).to(self._true_dtype)
        self.intermediate_decoder = IntermediateDecoder(embed_dim=1024).to(self._true_dtype)
        
        self.org_neck = copy.deepcopy(base_net.encoder.neck)
        self.org_out = copy.deepcopy(base_net.out)
        
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
        
        # Disconnect original components to prevent accidental usage
        self.base_net.encoder.neck = nn.Identity()
        self.base_net.out = nn.Identity()
        
        self.active_head = 'both'

        # ---> VOLCANO MERGER PARAMETERS <---
        self.learn_volcano = learn_volcano
        if self.learn_volcano:
            self.alpha = nn.Parameter(torch.tensor([float(alpha)], dtype=self._true_dtype))
            self.beta = nn.Parameter(torch.tensor([float(beta)], dtype=self._true_dtype))
            models_logger.info(f">// [VOLCANO MERGER] Learnable Mode Active (alpha={alpha}, beta={beta})")
        else:
            self.register_buffer('alpha', torch.tensor([float(alpha)], dtype=self._true_dtype))
            self.register_buffer('beta', torch.tensor([float(beta)], dtype=self._true_dtype))
            models_logger.info(f">>> [VOLCANO MERGER] Static Mode Locked (alpha={alpha}, beta={beta})")

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            self.intermediate_features[layer_idx] = output
        return hook

    def _reshape_to_spatial(self, x):
        """Converts arbitrary ViT outputs to standard spatial format (B, C, H, W)"""
        if x.ndim == 4 and x.shape[1] == 1024:
            return x
        if x.ndim == 4 and x.shape[-1] == 1024:
            return x.permute(0, 3, 1, 2).contiguous()
        if x.ndim == 3:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return self._true_dtype

    @dtype.setter
    def dtype(self, new_dtype):
        self._true_dtype = new_dtype
        self.to(new_dtype)

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def pixel_shuffle(self, x):
        """ Recreates Cellpose's hidden 8x upsampling reshape (192 -> 3 channels) """
        B, C, H, W = x.shape
        out_c = C // 64 
        x = x.view(B, out_c, 8, 8, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(B, out_c, H * 8, W * 8)

    def apply_volcano_merger(self, out_c):
        """ Dynamically blends probability gradients into the vector flows """
        import torchvision.transforms.functional as TF
        import torch.nn.functional as F
        
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
        self.intermediate_features.clear()
        x = x.to(self._true_dtype)
        
        # ========================================
        # CELL PATHWAY (TRAIN vs EVAL ROUTING)
        # ========================================
        if self.training:
            final_vit_out = self.base_net.encoder(x) 
            feat_c = self.cell_neck(final_vit_out)
            out_c = self.pixel_shuffle(self.cell_out(feat_c))
        else:
            # PASS 1: Base Map from Train Head
            vit_out_pass1 = self.base_net.encoder(x)
            feat_c_pass1 = self.cell_neck(vit_out_pass1)
            out_c_pass1 = self.pixel_shuffle(self.cell_out(feat_c_pass1))
            
            # Feedback Map Extraction
            prob_map = torch.sigmoid(out_c_pass1[:, 2:3, :, :])
            x_feedback = x * prob_map
            
            # Flush hooks for Pass 2 routing
            self.intermediate_features.clear()
            
            # PASS 2: Inference Pass
            final_vit_out = self.base_net.encoder(x_feedback)
            feat_c = self.cell_predict_neck(final_vit_out)
            out_c = self.pixel_shuffle(self.cell_predict_out(feat_c))
            
        out_c = self.apply_volcano_merger(out_c) 
        style_c = torch.mean(feat_c, dim=(2, 3))
        
        # ========================================
        # ORGANELLE PATHWAY 
        # ========================================
        # 1. Harvest and map all 7 structural layers to spatial shapes
        spatial_features = [self._reshape_to_spatial(self.intermediate_features[i]) for i in self.target_layers]
        
        # 2. Execute cross-layer attention fusion
        fused_early = self.layer_attention_concat(spatial_features)
        
        # 3. Interleaved processing loop (2x layers)
        transformed_early = self.organelle_transformer(fused_early)
        
        # 4. Standard upsampling down-projections
        decoded_early = self.intermediate_decoder(transformed_early)
        
        feat_o = self.org_neck(decoded_early)
        out_o = self.pixel_shuffle(self.org_out(feat_o))
        style_o = torch.mean(feat_o, dim=(2, 3))

        if self.active_head == 'cells':
            return out_c, style_c
        elif self.active_head == 'organelles':
            return out_o, style_o
        else:
            return (out_c, out_o), (style_c, style_o)


class CellposeModel():
    def __init__(self, gpu=False, pretrained_model="cpsam", custom_weights=None, model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True, manual=True, 
                 freeze_backbone=False, random=False, learn_volcano=True, alpha=1.0, beta=0.1, ASPP=False):

        if diam_mean is not None:
            models_logger.warning("diam_mean argument are not used in v4.0.1+. Ignoring this argument...")
        if model_type is not None:
            models_logger.warning("model_type argument is not used in v4.0.1+. Ignoring this argument...")
        
        if nchan is None:
            nchan = 3
            
        self.nchan = nchan
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
        
        base_net = Transformer(dtype=dtype).to(self.device)

        if not (custom_weights is not None and os.path.exists(custom_weights)):
            if os.path.exists(self.pretrained_model):
                models_logger.info(f">>>> loading base model {self.pretrained_model}")
                base_net.load_model(self.pretrained_model, device=self.device)
            else:
                if os.path.split(self.pretrained_model)[-1] != 'cpsam':
                    raise FileNotFoundError('model file not recognized')
                cache_CPSAM_model_path()
                base_net.load_model(self.pretrained_model, device=self.device)

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
                        half_weight = layer.weight.clone() / 2.0
                        new_conv.weight[:, :3, :, :] = half_weight
                        new_conv.weight[:, 3:, :, :] = half_weight
                        if layer.bias is not None:
                            new_conv.bias.copy_(layer.bias)
                            
                    setattr(patch_embed, name, new_conv)
                    getattr(patch_embed, name).to(dtype=dtype, device=self.device)
                    models_logger.info(f"Input {self.nchan}-channel modification complete!")
                    break

        if not manual or custom_weights is not None:
            models_logger.info("Injecting DualPathTransformer (Branching Cell at End, Organelles from Early Hooks)...")
            self.net = DualPathTransformer(
                base_net, 
                randomize_org=random,
                learn_volcano=learn_volcano,
                alpha=alpha,
                beta=beta,
                use_aspp=ASPP
            ).to(self.device)
        else:
            self.net = base_net

        self.freeze_backbone = freeze_backbone

        if custom_weights is not None and os.path.exists(custom_weights):
            models_logger.info(f">>>> loading CUSTOM post-architectural weights {custom_weights}")
            self.net.load_model(custom_weights, device=self.device)

        self.set_freeze_backbone(self.freeze_backbone)
        

    def set_freeze_backbone(self, freeze=True):
        self.freeze_backbone = freeze
        if freeze:
            models_logger.info("\n>>> [MODELS] FREEZING BACKBONE: Only the Dual Decoder Paths will be trained.")
        else:
            models_logger.info("\n>>> [MODELS] UNFREEZING BACKBONE: The entire network will be trained (End-to-End).")
            
        for name, param in self.net.named_parameters():
            if any(key in name for key in ['cell_predict_neck', 'cell_predict_out']):
                param.requires_grad = False 
            elif any(key in name for key in ['cell_neck', 'cell_out', 'org_neck', 'org_out', 'layer_attention_concat', 'organelle_transformer', 'intermediate_decoder']):
                param.requires_grad = True  
            elif name in ['alpha', 'beta']:
                param.requires_grad = getattr(self.net, 'learn_volcano', False)
            else:
                param.requires_grad = not freeze

        
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
        tic = time.time()
        shape = x.shape
        
        outputs = {}
        heads_to_run = ['cells', 'organelles'] if active_head == 'both' else [active_head]

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

        if hasattr(self.net, 'active_head'):
            self.net.active_head = active_head

        return outputs
    
    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
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
