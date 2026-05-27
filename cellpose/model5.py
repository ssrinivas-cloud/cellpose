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
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import gc
import cv2
import copy 
import logging
import matplotlib.pyplot as plt  
from huggingface_hub import HfApi

from cellpose import io, utils, dynamics
from cellpose.transforms import normalize_img, random_rotate_and_resize

# Import internal modules (assuming this is placed within the cellpose package)
from . import transforms, plot
from .vit_sam import Transformer
from .core import assign_device, run_net, run_3D

train_logger = logging.getLogger(__name__)
models_logger = logging.getLogger(__name__)

# =========================================================================
# TRAINING FUNCTIONS
# =========================================================================

def _loss_fn_class(lbl, y, class_weights=None):
    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    loss3 = criterion3(y[:, :-3], lbl[:, 0].long())
    return loss3

def total_variation_loss(pred_flows):
    # Penalizes the cell head if neighboring pixels point in wildly different directions
    diff_h = torch.abs(pred_flows[:, :, :, 1:] - pred_flows[:, :, :, :-1])
    diff_v = torch.abs(pred_flows[:, :, 1:, :] - pred_flows[:, :, :-1, :])
    return diff_h.mean() + diff_v.mean()

def dice_loss(pred_logits, target_mask, smooth=1e-5):
    # Apply sigmoid to get probabilities (0 to 1)
    pred_probs = torch.sigmoid(pred_logits)
    
    # Flatten tensors
    pred_flat = pred_probs.view(-1)
    target_flat = target_mask.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Calculate Dice coefficient and return loss
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def _loss_fn_seg(lbl, y, device, flow_weight=0.5, bce_weight=1.0, dice_weight=2.0, tv_weight=0.005):
    criterion_mse = nn.MSELoss(reduction="mean")
    criterion_bce = nn.BCEWithLogitsLoss(reduction="mean")
    
    veci = 5. * lbl[:, -2:] # GT Cell Flows
    pred_flows = y[:, -3:-1] # Predicted Cell Flows
    
    target_mask = (lbl[:, -3] > 0.5).to(y.dtype)
    pred_logits = y[:, -1]
    
    # 1. Calculate base losses
    loss_flow = criterion_mse(pred_flows, veci)
    loss_bce = criterion_bce(pred_logits, target_mask)
    loss_dice = dice_loss(pred_logits, target_mask)
    
    # 2. Calculate the smoothness penalty
    loss_tv = total_variation_loss(pred_flows)
    
    # 3. Combine: Force the model to expand shapes using Dice (dice_weight=2.0)
    loss_prob_total = (loss_bce * bce_weight) + (loss_dice * dice_weight)
    total_loss = loss_prob_total + (loss_flow * flow_weight) + (loss_tv * tv_weight)
    
    return total_loss, loss_prob_total, loss_flow, loss_tv

# =========================================================================
# CRITICAL UPDATE: Organelles now use the Hybrid Dice + BCE Loss
# =========================================================================
def _loss_fn_org(lbl, y, device, flow_weight=0.5, bce_weight=1.0, dice_weight=2.0):
    criterion_mse = nn.MSELoss(reduction="mean")
    criterion_bce = nn.BCEWithLogitsLoss(reduction="mean")
    
    veci = 5. * lbl[:, -2:] # GT Org Flows
    pred_flows = y[:, -3:-1] # Predicted Org Flows
    
    target_mask = (lbl[:, -3] > 0.5).to(y.dtype)
    pred_logits = y[:, -1]
    
    # 1. Calculate base losses
    loss_flow = criterion_mse(pred_flows, veci)
    loss_bce = criterion_bce(pred_logits, target_mask)
    loss_dice = dice_loss(pred_logits, target_mask)
    
    # 2. Combine (No TV loss needed for tiny organelles)
    loss_prob_total = (loss_bce * bce_weight) + (loss_dice * dice_weight)
    total_loss = loss_prob_total + (loss_flow * flow_weight)
    
    return total_loss, loss_prob_total, loss_flow, torch.tensor(0.0, device=device)

def _get_batch(inds, data=None, labels_c=None, labels_o=None, two_tail=False):
    imgs = []
    for i in inds:
        img = data[i].copy()
        
        # Standardize to (C, H, W)
        if img.ndim == 3 and img.shape[-1] <= 3:
            img = img.transpose(2, 0, 1)
            
        # Detect and drop fake zero-padded 3rd channel if it was added upstream
        if img.ndim == 3 and img.shape[0] == 3:
            if np.max(img[2]) == 0.0 and np.min(img[2]) == 0.0:
                img = img[:2]
                
        # ---> DYNAMIC BATCH MAPPING <---
        if img.ndim == 3 and img.shape[0] == 2:
            if two_tail:
                img = np.concatenate([
                    np.repeat(img[0:1], 3, axis=0),
                    np.repeat(img[1:2], 3, axis=0)
                ], axis=0)
            else:
                img = np.concatenate([img[0:1], img[1:2], np.zeros_like(img[0:1])], axis=0)
            
        img = normalize_img(img, axis=0) 
        imgs.append(img)
        
    lbls_c = [labels_c[i][1:] for i in inds] 
    lbls_o = [labels_o[i][1:] for i in inds]
    return imgs, lbls_c, lbls_o

def _process_train_test_paired(train_data, train_labels_c, train_labels_o, test_data, test_labels_c, test_labels_o, device=None):
    train_logger.info(">>> Computing Cell Flows...")
    train_flows_c = dynamics.labels_to_flows(train_labels_c, device=device)
    test_flows_c = dynamics.labels_to_flows(test_labels_c, device=device) if test_labels_c else None
    
    train_logger.info(">>> Computing Organelle Flows...")
    train_flows_o = dynamics.labels_to_flows(train_labels_o, device=device)
    test_flows_o = dynamics.labels_to_flows(test_labels_o, device=device) if test_labels_o else None
    
    nimg = len(train_data)
    nimg_test = len(test_data) if test_data else 0
    
    train_logger.info(">>> Computing Diameters...")
    diam_train = np.array([utils.diameters(train_flows_c[k][0])[0] for k in range(nimg)])
    diam_train[diam_train < 5] = 5.
    if test_data:
        diam_test = np.array([utils.diameters(test_flows_c[k][0])[0] for k in range(nimg_test)])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None
        
    return train_flows_c, train_flows_o, test_flows_c, test_flows_o, diam_train, diam_test


def train_seg(net, train_data=None, train_labels_c=None, train_labels_o=None,
              test_data=None, test_labels_c=None, test_labels_o=None,
              batch_size=8, learning_rate=3e-4, n_epochs=100, weight_decay=0.0001, 
              rescale=False, scale_range=0.5, bsize=256,
              model_name=None, class_weights=None, hf_repo_id=None, hf_token=None, 
              save_flows=False, visualize=False, debug=False, 
              unfreeze_backbone=50, test_result=10, normalize_loss=False, 
              turnoff_cell_loss=False, only_cell_loss=False, 
              two_tail=False, cell_loss_coeff=1.0, org_loss_coeff=1.0, **kwargs):
    
    device = net.device
    original_net_dtype = net.dtype 
    if net.dtype == torch.bfloat16:
        net.dtype = torch.float32

    out = _process_train_test_paired(train_data, train_labels_c, train_labels_o, test_data, test_labels_c, test_labels_o, device=device)
    train_flows_c, train_flows_o, test_flows_c, test_flows_o, diam_train, diam_test = out
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)
    nimg = len(train_data)
    nimg_test = len(test_data) if test_data else 0

    warmup_epochs = min(10, n_epochs // 10) 
    cosine_epochs = max(0, n_epochs - warmup_epochs)
    LR_warmup = np.linspace(0, learning_rate, warmup_epochs)
    min_lr = learning_rate * 0.01 
    LR_cosine = min_lr + 0.5 * (learning_rate - min_lr) * (1 + np.cos(np.pi * np.arange(cosine_epochs) / cosine_epochs)) if cosine_epochs > 0 else np.array([])
    LR = np.concatenate([LR_warmup, LR_cosine])
    
    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}, two_tail={two_tail}")
    train_logger.info(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")
    
    if turnoff_cell_loss and only_cell_loss:
        raise ValueError("You cannot have both 'turnoff_cell_loss' and 'only_cell_loss' set to True.")

    if turnoff_cell_loss:
        train_logger.info(">>> [ABLATION MODE] Cell Loss is turned OFF. Training Organelles only.")
    elif only_cell_loss:
        train_logger.info(">>> [ABLATION MODE] Organelle Loss is turned OFF. Training Cells only.")

    # =================================================================
    # LOSS NORMALIZATION LOGIC
    # =================================================================
    if normalize_loss:
        total_coeff = cell_loss_coeff + org_loss_coeff
        if total_coeff > 0:
            eff_cell_coeff = cell_loss_coeff / total_coeff
            eff_org_coeff = org_loss_coeff / total_coeff
            train_logger.info(f">>> [LOSS NORMALIZATION] Active. Effective Multipliers -> Cells: {eff_cell_coeff:.4f} | Orgs: {eff_org_coeff:.4f}")
        else:
            eff_cell_coeff = 0.5
            eff_org_coeff = 0.5
    else:
        eff_cell_coeff = cell_loss_coeff
        eff_org_coeff = org_loss_coeff

    # =================================================================
    # PARAMETER FREEZE LOGIC (Percentage-based Unfreezing)
    # =================================================================
    unfreeze_backbone = max(0, min(100, unfreeze_backbone))
    unfreeze_epoch = int((unfreeze_backbone / 100.0) * n_epochs)
    
    is_frozen = unfreeze_epoch > 0

    if is_frozen:
        if unfreeze_epoch >= n_epochs:
            train_logger.info(f"\n>>> [MODELS] Backbone will remain PERMANENTLY frozen (unfreeze_backbone={unfreeze_backbone}%).")
        else:
            train_logger.info(f"\n>>> [MODELS] Phase 1: Freezing ViT Backbone. Will unfreeze at Epoch {unfreeze_epoch} ({unfreeze_backbone}%).")
    else:
        train_logger.info("\n>>> [MODELS] Training Entire Dual-Path Network End-to-End from start (unfreeze_backbone=0%).")

    for name, param in net.named_parameters():
        if any(x in name for x in ['cell_predict_neck', 'cell_predict_out']):
            param.requires_grad = False
        elif any(x in name for x in ['cell_neck', 'cell_out', 'org_neck', 'org_out', 'layer_attention_concat', 'organelle_transformer', 'intermediate_decoder']):
            param.requires_grad = True
        else:
            param.requires_grad = not is_frozen

    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and net.dtype in [torch.float16, torch.bfloat16]))
    accumulation_steps = max(1, 8 // batch_size)

    if hasattr(net, 'active_head'):
        net.active_head = 'both'

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() / "models"
    save_path.mkdir(exist_ok=True)
    filename = save_path / model_name
    
    best_test_loss = float('inf')
    best_model_path = str(filename)
    
    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    
    for iepoch in range(n_epochs):
        rperm = np.random.permutation(nimg)
        for param_group in optimizer.param_groups: param_group["lr"] = LR[iepoch]
        
        net.train() 
        optimizer.zero_grad()

        for k in range(0, nimg, batch_size):
            kend = min(k + batch_size, nimg)
            inds = rperm[k:kend]
            
            imgs, lbls_c, lbls_o = _get_batch(inds, data=train_data, labels_c=train_flows_c, labels_o=train_flows_o, two_tail=two_tail)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(len(diams), "float32")
            
            lbls_stacked = [np.concatenate((lbls_c[i], lbls_o[i]), axis=0) for i in range(len(inds))]
            imgi, lbl_aug = random_rotate_and_resize(imgs, Y=lbls_stacked, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize))[:2]
            lbl_c_aug, lbl_o_aug = lbl_aug[:, :3, :, :], lbl_aug[:, 3:, :, :]
                                                                                                        
            X = torch.from_numpy(imgi).to(device)
            L_c = torch.from_numpy(lbl_c_aug).to(device)
            L_o = torch.from_numpy(lbl_o_aug).to(device)

            with torch.autocast(device_type=device.type, dtype=net.dtype):
                outputs, style = net(X) 
                y_cell, y_org = outputs
                
                if debug and k == 0:
                    train_logger.info(f"\n[DEBUG-SYS] --- EPOCH {iepoch} TENSOR & HARDWARE TELEMETRY ---")
                    
                    vram_alloc = torch.cuda.memory_allocated(device) / 1e9
                    vram_reserv = torch.cuda.memory_reserved(device) / 1e9
                    train_logger.info(f"[DEBUG-SYS] VRAM Usage -> Allocated: {vram_alloc:.2f} GB | Reserved: {vram_reserv:.2f} GB")
                    
                    train_logger.info(f"[DEBUG-SYS] Input Batch Shape: {X.shape} | Min: {X.min().item():.3f} | Max: {X.max().item():.3f} | Mean: {X.mean().item():.3f}")
                    
                    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    train_logger.info(f"[DEBUG-SYS] Trainable Params this step: {n_trainable:,}")

                    y_c_np = (y_cell[0, -1] > 0.0).detach().cpu().numpy()
                    y_o_np = (y_org[0, -1] > 0.0).detach().cpu().numpy()
                    pred_cell_blobs = scipy.ndimage.label(y_c_np)[1]
                    pred_org_blobs = scipy.ndimage.label(y_o_np)[1]
                    train_logger.info(f"[DEBUG-SYS] Dual-Head Prediction -> Cells: {pred_cell_blobs} | Orgs: {pred_org_blobs}")

                    try:
                        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                        img_c = imgi[0, 0] 
                        img_o = imgi[0, 3] if two_tail else imgi[0, 1] 
                        
                        gt_c = lbl_c_aug[0, 0] > 0
                        gt_o = lbl_o_aug[0, 0] > 0
                        
                        cell_prob_map = torch.sigmoid(y_cell[0, -1]).detach().cpu().numpy()
                        org_prob_map = torch.sigmoid(y_org[0, -1]).detach().cpu().numpy()
                        
                        c_flow_vis = np.clip((y_cell[0, -3:-1].detach().cpu().numpy().transpose(1, 2, 0) + 5) / 10, 0, 1)
                        o_flow_vis = np.clip((y_org[0, -3:-1].detach().cpu().numpy().transpose(1, 2, 0) + 5) / 10, 0, 1)
                        
                        c_flow_rgb = np.concatenate([c_flow_vis, np.zeros_like(c_flow_vis[..., :1])], axis=-1)
                        o_flow_rgb = np.concatenate([o_flow_vis, np.zeros_like(o_flow_vis[..., :1])], axis=-1)

                        axes[0, 0].imshow(img_c, cmap='gray')
                        if np.any(gt_c): axes[0, 0].contour(gt_c, colors='lime', linewidths=1.0)
                        if np.any(y_c_np): axes[0, 0].contour(y_c_np, colors='red', linewidths=1.0, linestyles='dashed')
                        axes[0, 0].set_title("Cells: Overlays")
                        axes[0, 0].axis('off')
                        
                        im_prob_c = axes[0, 1].imshow(cell_prob_map, cmap='magma', vmin=0, vmax=1)
                        axes[0, 1].set_title("Cells: Probability Map")
                        axes[0, 1].axis('off')
                        fig.colorbar(im_prob_c, ax=axes[0, 1], fraction=0.046, pad=0.04)

                        axes[0, 2].imshow(c_flow_rgb)
                        axes[0, 2].set_title("Cells: Flow Vectors")
                        axes[0, 2].axis('off')

                        axes[1, 0].imshow(img_o, cmap='gray')
                        if np.any(gt_o): axes[1, 0].contour(gt_o, colors='lime', linewidths=1.0)
                        if np.any(y_o_np): axes[1, 0].contour(y_o_np, colors='red', linewidths=1.0, linestyles='dashed')
                        axes[1, 0].set_title("Orgs: Overlays")
                        axes[1, 0].axis('off')
                        
                        im_prob_o = axes[1, 1].imshow(org_prob_map, cmap='magma', vmin=0, vmax=1)
                        axes[1, 1].set_title("Orgs: Probability Map")
                        axes[1, 1].axis('off')
                        fig.colorbar(im_prob_o, ax=axes[1, 1], fraction=0.046, pad=0.04)

                        axes[1, 2].imshow(o_flow_rgb)
                        axes[1, 2].set_title("Orgs: Flow Vectors")
                        axes[1, 2].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(save_path / f"debug_training_crop_epoch_{iepoch:04d}.png")
                        plt.close(fig)
                    except Exception as e:
                        train_logger.warning(f"Debug plotting failed: {e}")

                if only_cell_loss:
                    loss_cell, l_prob_c, l_flow_c, l_tv_c = _loss_fn_seg(L_c, y_cell, device)
                    loss = loss_cell / accumulation_steps
                    if debug and k == 0:
                        train_logger.info(f"[DEBUG-LOSS] CELL ONLY -> Total: {loss_cell.item():.4f} | Prob (BCE+Dice): {l_prob_c.item():.4f} | Flow: {l_flow_c.item():.4f} | TV: {l_tv_c.item():.4f}")
                elif turnoff_cell_loss:
                    loss_org, l_prob_o, l_flow_o, _ = _loss_fn_org(L_o, y_org, device) 
                    loss = loss_org / accumulation_steps
                    if debug and k == 0:
                        train_logger.info(f"[DEBUG-LOSS] ORG ONLY -> Total: {loss_org.item():.4f} | Prob: {l_prob_o.item():.4f} | Flow: {l_flow_o.item():.4f}")
                else:
                    loss_cell, l_prob_c, l_flow_c, l_tv_c = _loss_fn_seg(L_c, y_cell, device)
                    loss_org, l_prob_o, l_flow_o, _ = _loss_fn_org(L_o, y_org, device) 
                    loss = (eff_cell_coeff*loss_cell + eff_org_coeff*loss_org) / accumulation_steps
                    if debug and k == 0:
                        train_logger.info(f"[DEBUG-LOSS] DUAL HEAD -> Cell [Total:{loss_cell.item():.4f}, Prob:{l_prob_c.item():.4f}, Flow:{l_flow_c.item():.4f}] | Org [Total:{loss_org.item():.4f}]")

            scaler.scale(loss).backward()
            
            if (k // batch_size + 1) % accumulation_steps == 0 or (k + batch_size) >= nimg:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                if debug and k == 0:
                    train_logger.info(f"[DEBUG-SYS] Raw Gradient Norm (Before Clip): {total_norm.item():.4f}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss = (loss.item() * accumulation_steps) * len(imgi)
            lavg += train_loss
            nsum += len(imgi)
            train_losses[iepoch] += train_loss
            
        train_losses[iepoch] /= nimg

        if is_frozen and iepoch >= unfreeze_epoch and unfreeze_epoch < n_epochs:
            train_logger.info(f"\n>>> [AUTO-UNFREEZE] Milestone Reached (Epoch {iepoch}/{n_epochs} - {unfreeze_backbone}%). UNFREEZING WHOLE BACKBONE FOR FINE-TUNING!")
            
            newly_unfrozen_params = []
            for name, param in net.named_parameters(): 
                if not param.requires_grad and 'cell_predict' not in name:
                    param.requires_grad = True
                    newly_unfrozen_params.append(param)
            
            if newly_unfrozen_params:
                optimizer.add_param_group({'params': newly_unfrozen_params, 'lr': LR[iepoch], 'weight_decay': weight_decay})
                
            is_frozen = False

        if iepoch % test_result == 0 or iepoch == n_epochs - 1:
            lavgt = 0.
            if test_data:
                rperm_test = np.random.permutation(nimg_test)
                for ibatch in range(0, nimg_test, batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm_test[ibatch:ibatch + batch_size]
                        imgs, lbls_c, lbls_o = _get_batch(inds, data=test_data, labels_c=test_flows_c, labels_o=test_flows_o, two_tail=two_tail)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(len(diams), "float32")
                        
                        lbls_stacked = [np.concatenate((lbls_c[i], lbls_o[i]), axis=0) for i in range(len(inds))]
                        imgi, lbl_aug = random_rotate_and_resize(imgs, Y=lbls_stacked, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize))[:2]
                        lbl_c_aug, lbl_o_aug = lbl_aug[:, :3, :, :], lbl_aug[:, 3:, :, :]
                            
                        X = torch.from_numpy(imgi).to(device)
                        L_c = torch.from_numpy(lbl_c_aug).to(device)
                        L_o = torch.from_numpy(lbl_o_aug).to(device)

                        with torch.autocast(device_type=device.type, dtype=net.dtype):
                            outputs, style = net(X) 
                            y_cell, y_org = outputs
                            
                            if only_cell_loss:
                                loss_c, _, _, _ = _loss_fn_seg(L_c, y_cell, device)
                                loss = loss_c
                            elif turnoff_cell_loss:
                                loss_o, _, _, _ = _loss_fn_org(L_o, y_org, device) 
                                loss = loss_o
                            else:
                                loss_c, _, _, _ = _loss_fn_seg(L_c, y_cell, device)
                                loss_o, _, _, _ = _loss_fn_org(L_o, y_org, device) 
                                loss = eff_cell_coeff*loss_c + eff_org_coeff*loss_o
                        
                        lavgt += loss.item() * len(imgi)
                lavgt /= nimg_test
                test_losses[iepoch] = lavgt
                
                if lavgt < best_test_loss:
                    best_test_loss = lavgt
                    best_model_path = str(filename) + "_best"
                    net.save_model(best_model_path)
                    train_logger.info(f"*** New Best Model Saved! Test Loss: {best_test_loss:.4f} ***")

            if debug and test_data:
                # Use a specific import reference for the unified file evaluator
                temp_model_path = str(filename) + f"_eval_temp"
                net.save_model(temp_model_path)
                eval_model = CellposeModel(gpu=True, custom_weights=temp_model_path, use_bfloat16=False, nchan=(6 if two_tail else 3))
                
                masks_both, _, _ = eval_model.eval(test_data, batch_size=2, channels=[0,0], cellprob_threshold=0.0, rescale=1.0, active_head='both')
                pred_cells, pred_orgs = [m[0] for m in masks_both], [m[1] for m in masks_both]
                gt_cells, gt_orgs = [t[0] for t in test_flows_c], [t[0] for t in test_flows_o]
                
                n_cells_pred = pred_cells[0].max() if np.any(pred_cells[0]) else 0
                n_orgs_pred = pred_orgs[0].max() if np.any(pred_orgs[0]) else 0
                n_cells_gt = gt_cells[0].max() if np.any(gt_cells[0]) else 0
                n_orgs_gt = gt_orgs[0].max() if np.any(gt_orgs[0]) else 0
                
                train_logger.info(f"--- [DEBUG] Full Image Counts -> CELLS: Pred {n_cells_pred} (GT {n_cells_gt}) | ORGS: Pred {n_orgs_pred} (GT {n_orgs_gt}) ---")
                
                def calc_metrics(gt_masks, pred_masks):
                    tp = fp = fn = 0
                    for gt, pred in zip(gt_masks, pred_masks):
                        gt_bin, pred_bin = gt > 0, pred > 0
                        tp += np.logical_and(gt_bin, pred_bin).sum()
                        fp += np.logical_and(~gt_bin, pred_bin).sum()
                        fn += np.logical_and(gt_bin, ~pred_bin).sum()  
                    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                
                train_logger.info(f"--- [DEBUG] Test IOU -> CELLS: {calc_metrics(gt_cells, pred_cells):.4f} | ORGANELLES: {calc_metrics(gt_orgs, pred_orgs):.4f} ---")

                if visualize:
                    try:
                        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
                        fig.suptitle(f"Epoch {iepoch} - Post-Processed Mask Reconstructions", fontsize=18, y=0.98)
                        
                        img_disp = test_data[0].copy()
                        if img_disp.ndim == 3 and img_disp.shape[0] in [2, 3, 6]:
                            img_disp = img_disp.transpose(1, 2, 0)
                        
                        if img_disp.max() > img_disp.min(): 
                            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
                        
                        img_cell = img_disp[..., 0] 
                        img_org = img_disp[..., 3] if two_tail else (img_disp[..., 1] if img_disp.shape[-1] >= 2 else img_disp[..., 0])
                        
                        axes[0, 0].imshow(img_cell, cmap='gray')
                        if np.any(gt_cells[0]): axes[0, 0].contour(gt_cells[0] > 0, colors='lime', linewidths=1.2)
                        axes[0, 0].set_title(f"Cells - Actual GT ({n_cells_gt} masks)", fontsize=12)
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(img_cell, cmap='gray')
                        if np.any(pred_cells[0]): axes[0, 1].contour(pred_cells[0] > 0, colors='red', linewidths=1.2)
                        axes[0, 1].set_title(f"Cells - Predicted Reconstructed ({n_cells_pred} masks)", fontsize=12)
                        axes[0, 1].axis('off')
                        
                        axes[1, 0].imshow(img_org, cmap='gray')
                        if np.any(gt_orgs[0]): axes[1, 0].contour(gt_orgs[0] > 0, colors='lime', linewidths=1.2)
                        axes[1, 0].set_title(f"Organelles - Actual GT ({n_orgs_gt} masks)", fontsize=12)
                        axes[1, 0].axis('off')
                        
                        axes[1, 1].imshow(img_org, cmap='gray')
                        if np.any(pred_orgs[0]): axes[1, 1].contour(pred_orgs[0] > 0, colors='red', linewidths=1.2)
                        axes[1, 1].set_title(f"Organelles - Predicted Reconstructed ({n_orgs_pred} masks)", fontsize=12)
                        axes[1, 1].axis('off')
                        
                        plt.tight_layout()
                        vis_save_path = save_path / f"epoch_{iepoch:04d}_post_processed_overlay.png"
                        plt.savefig(vis_save_path, bbox_inches='tight', dpi=150)
                        plt.close(fig)
                        train_logger.info(f"[DEBUG] Post-processing visualization overlay written to: {vis_save_path}")
                    except Exception as e:
                        train_logger.warning(f"Post-processed overlay execution block errored: {e}")
                
                del eval_model
                torch.cuda.empty_cache()

        lavg /= nsum
        train_logger.info(f"Epoch {iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s")
        lavg, nsum = 0, 0

    net.save_model(str(filename))

    if original_net_dtype != torch.float32: net.dtype = original_net_dtype

    if hf_repo_id and hf_token:
        train_logger.info(f">>> Uploading BEST model ({best_model_path}) to Hugging Face Hub: {hf_repo_id}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, exist_ok=True)
        api.upload_file(
            path_or_fileobj=best_model_path, 
            path_in_repo=f"models/{model_name}_best", 
            repo_id=hf_repo_id, 
            repo_type="model"
        )
        train_logger.info(">>> Upload complete!")

    return best_model_path, train_losses, test_losses


# =========================================================================
# MODEL ARCHITECTURE
# =========================================================================

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


class CrossLayerAttentionConcat(nn.Module):
    """
    Treats the L different layers as a sequence of L tokens for each spatial pixel.
    Applies Self-Attention across the layers to let them communicate,
    then concatenates the attended features and fuses them.
    """
    def __init__(self, embed_dim=1024, num_layers=7, num_heads=8):
        super().__init__()
        self.num_layers = num_layers
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * num_layers, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        B, C, H, W = features[0].shape
        
        stacked = torch.stack(features, dim=1)
        x = stacked.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.num_layers, C)
        
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out 
        
        x = x.view(B, H, W, self.num_layers, C).permute(0, 3, 4, 1, 2)
        x_concat = x.reshape(B, self.num_layers * C, H, W)
        
        return self.fusion_conv(x_concat)


class TransformerBlock(nn.Module):
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
    def __init__(self, embed_dim=1024, depth=5, num_heads=16, mlp_ratio=4.0, use_aspp=False):
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
                for _ in range(depth - 1)
            ])
            
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        for i in range(self.depth):
            x_seq = x.view(B, C, -1).transpose(1, 2)
            x_seq = self.transformer_blocks[i](x_seq)
            x = x_seq.transpose(1, 2).view(B, C, H, W).contiguous()
            
            if self.use_aspp and i < (self.depth - 1):
                x = self.aspp_blocks[i](x)
                
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


class DualPathTransformer(nn.Module):
    def __init__(self, base_net, randomize_org=False, use_aspp=False):
        super().__init__()
        self.base_net = base_net
        self._true_dtype = next(base_net.parameters()).dtype
        
        self.diam_mean = getattr(base_net, 'diam_mean', nn.Parameter(torch.tensor([30.0], dtype=self._true_dtype)))
        self.diam_labels = getattr(base_net, 'diam_labels', nn.Parameter(torch.tensor([30.0], dtype=self._true_dtype)))
        
        self.target_layers = [0, 4, 8, 12, 16, 20, 23]
        self.intermediate_features = {}
        
        for layer_idx in self.target_layers:
            self.base_net.encoder.blocks[layer_idx].register_forward_hook(
                self._get_hook(layer_idx)
            )
            
        self.cell_neck = copy.deepcopy(base_net.encoder.neck)
        self.cell_out = copy.deepcopy(base_net.out)
        
        self.cell_predict_neck = copy.deepcopy(base_net.encoder.neck)
        self.cell_predict_out = copy.deepcopy(base_net.out)
        
        self.layer_attention_concat = CrossLayerAttentionConcat(embed_dim=1024, num_layers=len(self.target_layers)).to(self._true_dtype)
        self.organelle_transformer = HybridOrganelleTransformer(embed_dim=1024, depth=5, use_aspp=use_aspp).to(self._true_dtype)
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
        
        self.base_net.encoder.neck = nn.Identity()
        self.base_net.out = nn.Identity()
        
        self.active_head = 'both'

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            self.intermediate_features[layer_idx] = output
        return hook

    def _reshape_to_spatial(self, x):
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
        B, C, H, W = x.shape
        out_c = C // 64 
        x = x.view(B, out_c, 8, 8, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(B, out_c, H * 8, W * 8)

    def forward(self, x):
        self.intermediate_features.clear()
        x = x.to(self._true_dtype)
        
        if self.training:
            final_vit_out = self.base_net.encoder(x) 
            feat_c = self.cell_neck(final_vit_out)
            out_c = self.pixel_shuffle(self.cell_out(feat_c))
        else:
            with torch.no_grad():
                vit_out_pass1 = self.base_net.encoder(x)
                feat_c_pass1 = self.cell_neck(vit_out_pass1)
                out_c_pass1 = self.pixel_shuffle(self.cell_out(feat_c_pass1))
                
                prob_map = torch.sigmoid(out_c_pass1[:, 2:3, :, :])
                
                if prob_map.mean() < 0.1:
                    x_feedback = x
                else:
                    x_feedback = x * prob_map
            
            self.intermediate_features.clear()
            
            final_vit_out = self.base_net.encoder(x_feedback)
            feat_c = self.cell_predict_neck(final_vit_out)
            out_c = self.pixel_shuffle(self.cell_predict_out(feat_c))
            
        style_c = torch.mean(feat_c, dim=(2, 3))
        
        spatial_features = [self._reshape_to_spatial(self.intermediate_features[i]) for i in self.target_layers]
        fused_early = self.layer_attention_concat(spatial_features)
        transformed_early = self.organelle_transformer(fused_early)
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
                 freeze_backbone=False, random=False, ASPP=False):

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
                use_aspp=ASPP
            ).to(self.device)
        else:
            self.net = base_net

        self.freeze_backbone = freeze_backbone
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
