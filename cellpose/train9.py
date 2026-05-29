"""
Copyright © 2026 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
Updated for compatibility with model6.py (Dual-Parasite Swin-FPN Architecture).
"""

import time
import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt  
from cellpose import io, utils, dynamics, model5 
from cellpose.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import logging
from huggingface_hub import HfApi

train_logger = logging.getLogger(__name__)

# =========================================================================
# LOSS FUNCTIONS
# =========================================================================

def focal_loss_with_logits(pred_logits, target_mask, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss to combat severe class imbalance (Background vs Foreground).
    alpha: Weights the foreground vs background. 
    gamma: The focusing parameter. Higher values strictly penalize 'hard' examples.
    """
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, reduction='none')
    pt = torch.exp(-bce_loss) 
    focal_loss = alpha * (1 - pt)**gamma * bce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    return focal_loss.sum()

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
    
    veci = 5. * lbl[:, -2:] # GT Cell Flows
    pred_flows = y[:, -3:-1] # Predicted Cell Flows
    
    target_mask = (lbl[:, -3] > 0.5).to(y.dtype)
    pred_logits = y[:, -1]
    
    # 1. Calculate base losses (MSE for vectors, Focal for probability)
    loss_flow = criterion_mse(pred_flows, veci)
    loss_focal = focal_loss_with_logits(pred_logits, target_mask, alpha=0.25, gamma=2.0)
    loss_dice = dice_loss(pred_logits, target_mask)
    
    # 2. Calculate the smoothness penalty
    loss_tv = total_variation_loss(pred_flows)
    
    # 3. Combine: (Re-using the bce_weight argument to apply to Focal Loss)
    loss_prob_total = (loss_focal * bce_weight) + (loss_dice * dice_weight)
    total_loss = loss_prob_total + (loss_flow * flow_weight) + (loss_tv * tv_weight)
    
    return total_loss, loss_prob_total, loss_flow, loss_tv

def _loss_fn_org(lbl, y, device, flow_weight=0.5, bce_weight=1.0, dice_weight=2.0):
    criterion_mse = nn.MSELoss(reduction="mean")
    
    veci = 5. * lbl[:, -2:] # GT Org Flows
    pred_flows = y[:, -3:-1] # Predicted Org Flows
    
    target_mask = (lbl[:, -3] > 0.5).to(y.dtype)
    pred_logits = y[:, -1]
    
    # 1. Calculate base losses (Higher Gamma for tiny organelles)
    loss_flow = criterion_mse(pred_flows, veci)
    loss_focal = focal_loss_with_logits(pred_logits, target_mask, alpha=0.25, gamma=3.0)
    loss_dice = dice_loss(pred_logits, target_mask)
    
    # 2. Combine (No TV loss needed for tiny organelles)
    loss_prob_total = (loss_focal * bce_weight) + (loss_dice * dice_weight)
    total_loss = loss_prob_total + (loss_flow * flow_weight)
    
    return total_loss, loss_prob_total, loss_flow, torch.tensor(0.0, device=device)

# =========================================================================
# DATA PREPARATION
# =========================================================================

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

# =========================================================================
# MAIN TRAINING LOOP
# =========================================================================

def train_seg(net, train_data=None, train_labels_c=None, train_labels_o=None,
              test_data=None, test_labels_c=None, test_labels_o=None,
              batch_size=8, learning_rate=3e-4, n_epochs=100, weight_decay=0.0001, 
              rescale=False, scale_range=0.5, bsize=256,
              model_name=None, class_weights=None, hf_repo_id=None, hf_token=None, 
              save_flows=False, visualize=False, debug=False, 
              unfreeze_schedule=[20, 5], test_result=10, normalize_loss=False, 
              turnoff_cell_loss=False, only_cell_loss=False, 
              two_tail=False, cell_loss_coeff=1.0, org_loss_coeff=1.0, 
              use_cosine_annealing=False, **kwargs):
    
    device = net.device
    original_net_dtype = net.dtype 
    if net.dtype == torch.bfloat16:
        net.dtype = torch.float32

    out = _process_train_test_paired(train_data, train_labels_c, train_labels_o, test_data, test_labels_c, test_labels_o, device=device)
    train_flows_c, train_flows_o, test_flows_c, test_flows_o, diam_train, diam_test = out
    
    # Safe handling for diam_labels in case architecture dropped it
    if hasattr(net, 'diam_labels'):
        net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)
        
    nimg = len(train_data)
    nimg_test = len(test_data) if test_data else 0

    # =================================================================
    # LEARNING RATE SCHEDULE
    # =================================================================
    if use_cosine_annealing:
        train_logger.info(">>> Using Cosine Annealing LR Schedule")
        warmup_epochs = min(10, n_epochs // 10) 
        cosine_epochs = max(0, n_epochs - warmup_epochs)
        LR_warmup = np.linspace(0, learning_rate, warmup_epochs)
        min_lr = learning_rate * 0.01 
        LR_cosine = min_lr + 0.5 * (learning_rate - min_lr) * (1 + np.cos(np.pi * np.arange(cosine_epochs) / cosine_epochs)) if cosine_epochs > 0 else np.array([])
        LR = np.concatenate([LR_warmup, LR_cosine])
    else:
        train_logger.info(">>> Using Default Step-Decay LR Schedule")
        LR = np.linspace(0, learning_rate, 10) # 1. Linear warmup
        LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10))) # 2. Constant phase

        # 3. Step decay at the end
        if n_epochs > 300:
            LR = LR[:-100]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10)) # Halves the LR every 10 epochs
        elif n_epochs > 99:
            LR = LR[:-50]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(5))  # Halves the LR every 5 epochs
    
    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}, two_tail={two_tail}")
    train_logger.info(f">>> AdamW, max learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")
    
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
    # PARAMETER FREEZE LOGIC (Cyclic Freeze/Unfreeze)
    # =================================================================
    if unfreeze_schedule is not None and len(unfreeze_schedule) == 2:
        cycle_interval, unfreeze_duration = unfreeze_schedule
        train_logger.info(f"\n>>> [MODELS] CYCLIC FREEZE SCHEDULE ACTIVE: Backbone will freeze for {cycle_interval - unfreeze_duration} epochs, then unfreeze for {unfreeze_duration} epochs.")
        start_frozen = True
    else:
        train_logger.info("\n>>> [MODELS] Training Entire Dual-Path Network End-to-End from start.")
        start_frozen = False

    for name, param in net.named_parameters():
        if any(x in name for x in ['cell_decoder', 'org_decoder']):
            param.requires_grad = True # Decoders ALWAYS train
        elif 'backbone' in name:
            param.requires_grad = not start_frozen # Start in the correct state
        else:
            param.requires_grad = True

    # Pass ALL parameters to the optimizer. We will simply toggle requires_grad on/off. 
    # AdamW will safely ignore parameters when their requires_grad is False.
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and net.dtype in [torch.float16, torch.bfloat16]))
    accumulation_steps = max(1, 8 // batch_size)

    if hasattr(net, 'active_head'):
        net.active_head = 'both'

    t0 = time.time()
    model_name = f"swincell_{t0}" if model_name is None else model_name
    save_path = Path.cwd() / "models"
    save_path.mkdir(exist_ok=True)
    filename = save_path / model_name
    
    best_test_loss = float('inf')
    best_model_path = str(filename)
    
    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    
    for iepoch in range(n_epochs):
        
        # =================================================================
        # CYCLIC AUTO-UNFREEZE / FREEZE TRIGGER 
        # =================================================================
        if unfreeze_schedule is not None and len(unfreeze_schedule) == 2:
            cycle_interval, unfreeze_duration = unfreeze_schedule
            if iepoch > 0:
                mod_epoch = iepoch % cycle_interval
                
                # Trigger Unfreeze (e.g. at epoch 20, 40, 60...)
                if mod_epoch == 0:
                    train_logger.info(f"\n>>> [AUTO-UNFREEZE] Epoch {iepoch}: UNFREEZING backbone for the next {unfreeze_duration} epochs.")
                    for name, param in net.named_parameters():
                        if 'backbone' in name:
                            param.requires_grad = True
                            
                # Trigger Freeze (e.g. at epoch 25, 45, 65...)
                elif mod_epoch == unfreeze_duration:
                    train_logger.info(f"\n>>> [AUTO-FREEZE] Epoch {iepoch}: FREEZING backbone.")
                    for name, param in net.named_parameters():
                        if 'backbone' in name:
                            param.requires_grad = False
                            param.grad = None # Safely clear lingering gradients
        
        rperm = np.random.permutation(nimg)
        for param_group in optimizer.param_groups: param_group["lr"] = LR[iepoch]
        
        net.train()
        optimizer.zero_grad()

        for k in range(0, nimg, batch_size):
            kend = min(k + batch_size, nimg)
            inds = rperm[k:kend]
            
            imgs, lbls_c, lbls_o = _get_batch(inds, data=train_data, labels_c=train_flows_c, labels_o=train_flows_o, two_tail=two_tail)
            diams = np.array([diam_train[i] for i in inds])
            
            # Safe handling if diam_mean is deprecated in new model structure
            net_diam = getattr(net, 'diam_mean', None)
            net_diam_val = net_diam.item() if net_diam is not None else 30.0
            rsc = diams / net_diam_val if rescale else np.ones(len(diams), "float32")
            
            lbls_stacked = [np.concatenate((lbls_c[i], lbls_o[i]), axis=0) for i in range(len(inds))]
            imgi, lbl_aug = random_rotate_and_resize(imgs, Y=lbls_stacked, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize))[:2]
            lbl_c_aug, lbl_o_aug = lbl_aug[:, :3, :, :], lbl_aug[:, 3:, :, :]
                                                                                                                                    
            X = torch.from_numpy(imgi).to(device)
            L_c = torch.from_numpy(lbl_c_aug).to(device)
            L_o = torch.from_numpy(lbl_o_aug).to(device)

            with torch.autocast(device_type=device.type, dtype=net.dtype):
                outputs, style = net(X) 
                
                # Dynamic Unpacking Logic to prevent ValueError
                if len(outputs) == 3:
                    y_cell, y_org, y_cell_pred = outputs
                elif len(outputs) == 2:
                    y_cell, y_org = outputs
                    y_cell_pred = y_cell  # Fallback
                else:
                    raise ValueError(f"Expected 2 or 3 outputs from model, got {len(outputs)}")
                
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
                        img_o = imgi[0, 3] if two_tail else imgi[0, 1] 
                        
                        gt_c = lbl_c_aug[0, 0] > 0
                        gt_o = lbl_o_aug[0, 0] > 0
                        
                        cell_prob_map = torch.sigmoid(y_cell[0, -1]).detach().cpu().numpy()
                        org_prob_map = torch.sigmoid(y_org[0, -1]).detach().cpu().numpy()
                        
                        # Plotting cell flows (using y_cell_pred, though y_cell now has flows too)
                        c_flow_vis = np.clip((y_cell_pred[0, -3:-1].detach().cpu().numpy().transpose(1, 2, 0) + 5) / 10, 0, 1)
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
                        plt.show()  # Display directly in Jupyter Notebook
                    except Exception as e:
                        train_logger.warning(f"Debug plotting failed: {e}")

                # ==========================================================
                # LOSS CALCULATIONS
                # ==========================================================
                # 1. CELL HEAD LOSS: Both Prob & Flow
                loss_cell, l_prob_c, l_flow_c, l_tv_c = _loss_fn_seg(L_c, y_cell, device)
                
                # 2. CELL PREDICT HEAD LOSS: Both Prob & Flow
                loss_cell_pred, l_prob_cp, l_flow_cp, l_tv_cp = _loss_fn_seg(L_c, y_cell_pred, device)
                
                # 3. ORGANELLE HEAD LOSS: Both Prob & Flow
                loss_org, l_prob_o, l_flow_o, _ = _loss_fn_org(L_o, y_org, device) 

                if only_cell_loss:
                    loss = (loss_cell + loss_cell_pred) / accumulation_steps
                    if debug and k == 0:
                        train_logger.info(f"[DEBUG-LOSS] CELL ONLY -> Cell Head Total: {loss_cell.item():.4f} | Cell Predict Total: {loss_cell_pred.item():.4f}")
                elif turnoff_cell_loss:
                    loss = loss_org / accumulation_steps
                    if debug and k == 0:
                        train_logger.info(f"[DEBUG-LOSS] ORG ONLY -> Total: {loss_org.item():.4f} | Prob: {l_prob_o.item():.4f} | Flow: {l_flow_o.item():.4f}")
                else:
                    total_cell_branch = loss_cell + loss_cell_pred
                    loss = (eff_cell_coeff * total_cell_branch + eff_org_coeff * loss_org) / accumulation_steps
                    if debug and k == 0:
                        train_logger.info(f"[DEBUG-LOSS] DUAL HEAD -> Cell [Head Total:{loss_cell.item():.4f}, Predict Total:{loss_cell_pred.item():.4f}] | Org [Total:{loss_org.item():.4f}]")


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
                        
                        net_diam = getattr(net, 'diam_mean', None)
                        net_diam_val = net_diam.item() if net_diam is not None else 30.0
                        rsc = diams / net_diam_val if rescale else np.ones(len(diams), "float32")
                        
                        lbls_stacked = [np.concatenate((lbls_c[i], lbls_o[i]), axis=0) for i in range(len(inds))]
                        imgi, lbl_aug = random_rotate_and_resize(imgs, Y=lbls_stacked, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize))[:2]
                        lbl_c_aug, lbl_o_aug = lbl_aug[:, :3, :, :], lbl_aug[:, 3:, :, :]
                            
                        X = torch.from_numpy(imgi).to(device)
                        L_c = torch.from_numpy(lbl_c_aug).to(device)
                        L_o = torch.from_numpy(lbl_o_aug).to(device)

                        with torch.autocast(device_type=device.type, dtype=net.dtype):
                            outputs, style = net(X) 
                            
                            if len(outputs) == 3:
                                y_cell, y_org, y_cell_pred = outputs
                            elif len(outputs) == 2:
                                y_cell, y_org = outputs
                                y_cell_pred = y_cell
                                
                            loss_cell, _, _, _ = _loss_fn_seg(L_c, y_cell, device)
                            loss_cell_pred, _, _, _ = _loss_fn_seg(L_c, y_cell_pred, device)
                            loss_org, _, _, _ = _loss_fn_org(L_o, y_org, device) 
                            
                            if only_cell_loss:
                                loss = loss_cell + loss_cell_pred
                            elif turnoff_cell_loss:
                                loss = loss_org
                            else:
                                total_cell_branch = loss_cell + loss_cell_pred
                                loss = eff_cell_coeff * total_cell_branch + eff_org_coeff * loss_org
                        
                        lavgt += loss.item() * len(imgi)
                lavgt /= nimg_test
                test_losses[iepoch] = lavgt
                
                # --- TRACK & SAVE BEST MODEL ---
                if lavgt < best_test_loss:
                    best_test_loss = lavgt
                    best_model_path = str(filename) + "_best"
                    net.save_model(best_model_path)
                    train_logger.info(f"*** New Best Model Saved! Test Loss: {best_test_loss:.4f} ***")

            # =====================================================================
            # FULL IMAGE POST-PROCESSING EVALUATION & VISUALIZATION 
            # =====================================================================
            if debug and test_data:
                temp_model_path = str(filename) + f"_eval_temp"
                net.save_model(temp_model_path)
                
                # Updated to use model6.CellposeModel wrapper
                eval_model = model5.CellposeModel(gpu=True, custom_weights=temp_model_path, use_bfloat16=False, nchan=(6 if two_tail else 3))
                
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
                        plt.show()  # Display directly in Jupyter Notebook
                    except Exception as e:
                        train_logger.warning(f"Post-processed overlay execution block errored: {e}")
                
                del eval_model
                torch.cuda.empty_cache()

        lavg /= nsum
        train_logger.info(f"Epoch {iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s")
        lavg, nsum = 0, 0

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
