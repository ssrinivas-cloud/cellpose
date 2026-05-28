"""
Copyright © 2026 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import time
import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt  
from cellpose import io, utils, dynamics, model3
from cellpose.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
import logging
from huggingface_hub import HfApi

# Ensure we are pulling your exact 3-head architecture


train_logger = logging.getLogger(__name__)

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
        
        # Standardize 2D grayscale to 3D (1, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        
        # Standardize to (C, H, W)
        if img.ndim == 3 and img.shape[-1] <= 3:
            img = img.transpose(2, 0, 1)
            
        # Detect and drop fake zero-padded 3rd channel if it was added upstream
        if img.ndim == 3 and img.shape[0] == 3:
            if np.max(img[2]) == 0.0 and np.min(img[2]) == 0.0:
                img = img[:2]
                
        # ---> DYNAMIC BATCH MAPPING FOR 1-CHANNEL <---
        if img.ndim == 3 and img.shape[0] == 1:
            if two_tail:
                # Replicate the single channel across both tails (6 channels total)
                img = np.repeat(img, 6, axis=0) 
            else:
                # Map to channel 1, pad channels 2 and 3 with zeros
                img = np.concatenate([img, np.zeros_like(img), np.zeros_like(img)], axis=0)
                
        # ---> DYNAMIC BATCH MAPPING FOR 2-CHANNEL <---
        elif img.ndim == 3 and img.shape[0] == 2:
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
              save_flows=False, visualize=False, debug=False, auto_unfreeze=False, 
              keep_encoder_frozen=False, turnoff_cell_loss=False, only_cell_loss=False, 
              two_tail=False, cell_loss_coeff=1, org_loss_coeff=1, test_interval=10, **kwargs):
    
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
    
    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}, two_tail={two_tail}, test_interval={test_interval}")
    train_logger.info(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")
    
    if turnoff_cell_loss and only_cell_loss:
        raise ValueError("You cannot have both 'turnoff_cell_loss' and 'only_cell_loss' set to True.")

    if turnoff_cell_loss:
        train_logger.info(">>> [ABLATION MODE] Cell Loss is turned OFF. Training Organelles only.")
    elif only_cell_loss:
        train_logger.info(">>> [ABLATION MODE] Organelle Loss is turned OFF. Training Cells only.")

    # =================================================================
    # PARAMETER FREEZE LOGIC
    # =================================================================
    is_frozen = False
    if auto_unfreeze and not keep_encoder_frozen:
        is_frozen = True
        train_logger.info("\n>>> [MODELS] Phase 1: Freezing ViT Backbone. Training Deep Dual Branches Only.")
        for name, param in net.named_parameters():
            if any(x in name for x in ['cell_neck', 'cell_out', 'org_neck', 'org_out']):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif keep_encoder_frozen:
        train_logger.info("\n>>> [MODELS] `keep_encoder_frozen` is True: Encoder and cell_predict head will remain PERMANENTLY frozen.")
        for name, param in net.named_parameters():
            # Strictly allow ONLY the training heads to receive gradients
            if any(x in name for x in ['cell_neck', 'cell_out', 'org_neck', 'org_out']):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        train_logger.info("\n>>> [MODELS] Training Entire Dual-Path Network End-to-End.")
        for name, param in net.named_parameters(): 
            if 'cell_predict' not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False # Explicitly ensure cell_predict is ALWAYS frozen

    # Safely construct the optimizer with ONLY parameters that require gradients
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
    
    # Best Model Tracking Initialization
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
                
                # ==========================================================
                # TRAINING PIPELINE SYSTEM & TENSOR TELEMETRY 
                # ==========================================================
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

                    # EXPANDED CROP-LEVEL VISUALIZATION BLOCK (2x3 Grid)
                    try:
                        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                        img_c = imgi[0, 0] 
                        
                        # Fix for single-channel handling in debug plotting
                        if two_tail and imgi.shape[1] >= 4:
                            img_o = imgi[0, 3]
                        elif imgi.shape[1] >= 2:
                            img_o = imgi[0, 1]
                        else:
                            img_o = imgi[0, 0] # Fallback to the primary channel
                        
                        gt_c = lbl_c_aug[0, 0] > 0
                        gt_o = lbl_o_aug[0, 0] > 0
                        
                        # Extract Raw Probabilities (Sigmoid) and Flows
                        cell_prob_map = torch.sigmoid(y_cell[0, -1]).detach().cpu().numpy()
                        org_prob_map = torch.sigmoid(y_org[0, -1]).detach().cpu().numpy()
                        
                        # Normalize flows purely for visualization (-5 to 5 mapped to 0 to 1)
                        c_flow_vis = np.clip((y_cell[0, -3:-1].detach().cpu().numpy().transpose(1, 2, 0) + 5) / 10, 0, 1)
                        o_flow_vis = np.clip((y_org[0, -3:-1].detach().cpu().numpy().transpose(1, 2, 0) + 5) / 10, 0, 1)
                        
                        # Pad with a blank blue channel for RGB visualization
                        c_flow_rgb = np.concatenate([c_flow_vis, np.zeros_like(c_flow_vis[..., :1])], axis=-1)
                        o_flow_rgb = np.concatenate([o_flow_vis, np.zeros_like(o_flow_vis[..., :1])], axis=-1)

                        # --- ROW 1: CELLS ---
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

                        # --- ROW 2: ORGANELLES ---
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
                    loss = (cell_loss_coeff*loss_cell + org_loss_coeff*loss_org) / accumulation_steps
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

        # =================================================================
        # AUTO-UNFREEZE TRIGGER
        # =================================================================
        if auto_unfreeze and is_frozen and iepoch >= 8:
            if keep_encoder_frozen:
                train_logger.info(f"\n>>> [AUTO-UNFREEZE] Skipped milestone. `keep_encoder_frozen` is True.")
                is_frozen = False # Set to false so it doesn't keep triggering this block
            else:
                train_logger.info(f"\n>>> [AUTO-UNFREEZE] 50% Milestone Reached ({iepoch}/{n_epochs}). UNFREEZING WHOLE BACKBONE FOR FINE-TUNING!")
                
                newly_unfrozen_params = []
                for name, param in net.named_parameters(): 
                    # Keep the cascaded head strictly frozen
                    if not param.requires_grad and 'cell_predict' not in name:
                        param.requires_grad = True
                        newly_unfrozen_params.append(param)
                
                if newly_unfrozen_params:
                    optimizer.add_param_group({'params': newly_unfrozen_params, 'lr': LR[iepoch], 'weight_decay': weight_decay})
                    
                is_frozen = False

        # Use the dynamic interval argument provided, or force a run on the last epoch
        if iepoch % test_interval == 0 or iepoch == n_epochs - 1:
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
                                loss = cell_loss_coeff*loss_c + org_loss_coeff*loss_o
                        
                        lavgt += loss.item() * len(imgi)
                lavgt /= nimg_test
                test_losses[iepoch] = lavgt
                
                # --- TRACK & SAVE BEST MODEL ---
                if lavgt < best_test_loss:
                    best_test_loss = lavgt
                    best_model_path = str(filename) + "_best"
                    net.save_model(best_model_path)
                    train_logger.info(f"*** New Best Model Saved! Test Loss: {best_test_loss:.4f} ***")

        lavg /= nsum
        train_logger.info(f"Epoch {iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s")
        lavg, nsum = 0, 0

        # =====================================================================
        # FULL IMAGE POST-PROCESSING EVALUATION & VISUALIZATION
        # =====================================================================
        if debug and test_data and (iepoch % test_interval == 0 or iepoch == n_epochs - 1):
            temp_model_path = str(filename) + f"_eval_temp"
            net.save_model(temp_model_path)
            eval_nchan = 6 if two_tail else 3
            eval_model = model3.CellposeModel(gpu=True, custom_weights=temp_model_path, use_bfloat16=False, nchan=eval_nchan)
            
            # PRE-FORMAT FOR EVAL: Cellpose's internal eval doesn't auto-pad to 3/6 channels for custom architectures.
            test_data_eval = []
            for img in test_data:
                img_eval = img.copy()
                # If 2D (Y, X), add channel dim -> (Y, X, 1)
                if img_eval.ndim == 2:
                    img_eval = img_eval[..., np.newaxis]
                # If (C, Y, X), transpose -> (Y, X, C) for Cellpose eval
                elif img_eval.ndim == 3 and img_eval.shape[0] <= 3:
                    img_eval = img_eval.transpose(1, 2, 0)
                    
                # Pad dynamically to match model's expected nchan
                if img_eval.shape[-1] == 1:
                    if two_tail:
                        img_eval = np.repeat(img_eval, 6, axis=-1)
                    else:
                        img_eval = np.concatenate([img_eval, np.zeros_like(img_eval), np.zeros_like(img_eval)], axis=-1)
                elif img_eval.shape[-1] == 2:
                    if two_tail:
                        img_eval = np.concatenate([np.repeat(img_eval[..., 0:1], 3, axis=-1), np.repeat(img_eval[..., 1:2], 3, axis=-1)], axis=-1)
                    else:
                        img_eval = np.concatenate([img_eval, np.zeros_like(img_eval[..., 0:1])], axis=-1)
                        
                test_data_eval.append(img_eval)

            # Pass channels=None so it doesn't try to strip our manually padded channels
            masks_both, _, _ = eval_model.eval(test_data_eval, batch_size=2, channels=None, cellprob_threshold=0.0, rescale=1.0, active_head='both')
            pred_cells, pred_orgs = [m[0] for m in masks_both], [m[1] for m in masks_both]
            gt_cells, gt_orgs = [t[0] for t in test_flows_c], [t[0] for t in test_flows_o]
            
            # Log the counts for the first image purely as a sys check
            n_cells_pred_0 = pred_cells[0].max() if np.any(pred_cells[0]) else 0
            n_orgs_pred_0 = pred_orgs[0].max() if np.any(pred_orgs[0]) else 0
            n_cells_gt_0 = gt_cells[0].max() if np.any(gt_cells[0]) else 0
            n_orgs_gt_0 = gt_orgs[0].max() if np.any(gt_orgs[0]) else 0
            train_logger.info(f"--- [DEBUG] Image 0 Counts -> CELLS: Pred {n_cells_pred_0} (GT {n_cells_gt_0}) | ORGS: Pred {n_orgs_pred_0} (GT {n_orgs_gt_0}) ---")
            
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
                    num_to_plot = min(3, len(test_data))
                    for img_idx in range(num_to_plot):
                        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
                        fig.suptitle(f"Epoch {iepoch} - Post-Processed Mask Reconstructions (Image {img_idx})", fontsize=18, y=0.98)
                        
                        img_disp = test_data[img_idx].copy()
                        if img_disp.ndim == 3 and img_disp.shape[0] in [1, 2, 3, 6]:
                            img_disp = img_disp.transpose(1, 2, 0)
                        elif img_disp.ndim == 2:
                            img_disp = img_disp[..., np.newaxis]
                        
                        if img_disp.max() > img_disp.min(): 
                            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
                        
                        # Fix for single-channel evaluation visualization
                        img_cell = img_disp[..., 0] 
                        img_org = img_disp[..., 3] if (two_tail and img_disp.shape[-1] >= 4) else (img_disp[..., 1] if img_disp.shape[-1] >= 2 else img_disp[..., 0])
                        
                        curr_gt_c = gt_cells[img_idx]
                        curr_pred_c = pred_cells[img_idx]
                        curr_gt_o = gt_orgs[img_idx]
                        curr_pred_o = pred_orgs[img_idx]

                        n_curr_cells_gt = curr_gt_c.max() if np.any(curr_gt_c) else 0
                        n_curr_cells_pred = curr_pred_c.max() if np.any(curr_pred_c) else 0
                        n_curr_orgs_gt = curr_gt_o.max() if np.any(curr_gt_o) else 0
                        n_curr_orgs_pred = curr_pred_o.max() if np.any(curr_pred_o) else 0

                        axes[0, 0].imshow(img_cell, cmap='gray')
                        if np.any(curr_gt_c): axes[0, 0].contour(curr_gt_c > 0, colors='lime', linewidths=1.2)
                        axes[0, 0].set_title(f"Cells - Actual GT ({n_curr_cells_gt} masks)", fontsize=12)
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(img_cell, cmap='gray')
                        if np.any(curr_pred_c): axes[0, 1].contour(curr_pred_c > 0, colors='red', linewidths=1.2)
                        axes[0, 1].set_title(f"Cells - Predicted Reconstructed ({n_curr_cells_pred} masks)", fontsize=12)
                        axes[0, 1].axis('off')
                        
                        axes[1, 0].imshow(img_org, cmap='gray')
                        if np.any(curr_gt_o): axes[1, 0].contour(curr_gt_o > 0, colors='lime', linewidths=1.2)
                        axes[1, 0].set_title(f"Organelles - Actual GT ({n_curr_orgs_gt} masks)", fontsize=12)
                        axes[1, 0].axis('off')
                        
                        axes[1, 1].imshow(img_org, cmap='gray')
                        if np.any(curr_pred_o): axes[1, 1].contour(curr_pred_o > 0, colors='red', linewidths=1.2)
                        axes[1, 1].set_title(f"Organelles - Predicted Reconstructed ({n_curr_orgs_pred} masks)", fontsize=12)
                        axes[1, 1].axis('off')
                        
                        plt.tight_layout()
                        vis_save_path = save_path / f"epoch_{iepoch:04d}_img_{img_idx}_post_processed_overlay.png"
                        plt.savefig(vis_save_path, bbox_inches='tight', dpi=150)
                        plt.close(fig)
                    train_logger.info(f"[DEBUG] Post-processing visualization overlays written to: {save_path}")
                except Exception as e:
                    train_logger.warning(f"Post-processed overlay execution block errored: {e}")
            
            del eval_model
            torch.cuda.empty_cache()

    # Final safeguard save
    net.save_model(str(filename))

    if original_net_dtype != torch.float32: net.dtype = original_net_dtype

    # --- HUGGING FACE BEST MODEL PUSH ---
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
