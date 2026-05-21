"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import time
import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt 
from cellpose import io, utils, model2, dynamics
from cellpose.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
import logging
from huggingface_hub import HfApi

train_logger = logging.getLogger(__name__)

def _loss_fn_class(lbl, y, class_weights=None):
    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    loss3 = criterion3(y[:, :-3], lbl[:, 0].long())
    return loss3

# --- STANDARD CELL LOSS ---
def _loss_fn_seg(lbl, y, device):
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, -2:]
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2. 
    loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    return loss + loss2

# --- STANDARD ORGANELLE LOSS ---
def _loss_fn_org(lbl, y, device):
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, -2:]
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2. 
    loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    return loss + loss2

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
              save_flows=False, visualize=False, debug=False, auto_unfreeze=False, 
              turnoff_cell_loss=False, only_cell_loss=False, two_tail=False, 
              cell_loss_coeff=1, org_loss_coeff=1, **kwargs):
    
    device = net.device
    original_net_dtype = net.dtype 
    if net.dtype == torch.bfloat16:
        net.dtype = torch.float32

    out = _process_train_test_paired(train_data, train_labels_c, train_labels_o, test_data, test_labels_c, test_labels_o, device=device)
    train_flows_c, train_flows_o, test_flows_c, test_flows_o, diam_train, diam_test = out
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)
    nimg = len(train_data)
    nimg_test = len(test_data) if test_data else 0

    # ---> RESTORED WARMUP + SLOW LINEAR DECAY <---
    # 1. Dedicate the first 10% of epochs (max 10) to safely warming up the optimizer
    warmup_epochs = min(10, n_epochs // 10) 
    decay_epochs = max(0, n_epochs - warmup_epochs)
    
    # Start at 0 and slowly climb to your target learning_rate
    LR_warmup = np.linspace(0, learning_rate, warmup_epochs) if warmup_epochs > 0 else np.array([])
    
    # 2. Strictly decrease from learning_rate down to 1% of learning_rate over the remaining epochs
    min_lr = learning_rate * 0.01 
    LR_decay = np.linspace(learning_rate, min_lr, decay_epochs) if decay_epochs > 0 else np.array([])
    
    # Combine the schedules
    LR = np.concatenate([LR_warmup, LR_decay])
    
    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}, two_tail={two_tail}")
    train_logger.info(f">>> AdamW, Peak LR={learning_rate:0.6f}, Final LR={min_lr:0.6f}, weight_decay={weight_decay:0.5f}")
    
    # --- ABLATION CONFLICT CHECK & LOGGING ---
    if turnoff_cell_loss and only_cell_loss:
        raise ValueError("You cannot have both 'turnoff_cell_loss' and 'only_cell_loss' set to True.")

    if turnoff_cell_loss:
        train_logger.info(">>> [ABLATION MODE] Cell Loss is turned OFF. Training Organelles only.")
    elif only_cell_loss:
        train_logger.info(">>> [ABLATION MODE] Organelle Loss is turned OFF. Training Cells only.")

    is_frozen = False
    if auto_unfreeze:
        is_frozen = True
        train_logger.info("\n>>> [MODELS] Phase 1: Freezing ViT Backbone. Training Deep Dual Branches Only.")
        for name, param in net.named_parameters():
            if any(x in name for x in ['cell_neck', 'cell_out', 'org_neck', 'org_out']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable_params = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=LR[0], weight_decay=weight_decay)
    else:
        train_logger.info("\n>>> [MODELS] Training Entire Dual-Path Network End-to-End.")
        for param in net.parameters(): 
            param.requires_grad = True
        optimizer = torch.optim.AdamW(net.parameters(), lr=LR[0], weight_decay=weight_decay)
        
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and net.dtype in [torch.float16, torch.bfloat16]))
    accumulation_steps = max(1, 8 // batch_size)

    if hasattr(net, 'active_head'):
        net.active_head = 'both'

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() / "models"
    save_path.mkdir(exist_ok=True)
    filename = save_path / model_name
    
    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    
    for iepoch in range(n_epochs):
        rperm = np.random.permutation(nimg)
        # Apply the current epoch's learning rate (warmup -> steady decay)
        for param_group in optimizer.param_groups: 
            param_group["lr"] = LR[iepoch]
        
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
                
                # ---> DYNAMIC LOSS TOGGLE (TRAINING) <---
                if only_cell_loss:
                    loss = _loss_fn_seg(L_c, y_cell, device) / accumulation_steps
                elif turnoff_cell_loss:
                    loss = _loss_fn_org(L_o, y_org, device) / accumulation_steps
                else:
                    loss_cell = _loss_fn_seg(L_c, y_cell, device)
                    loss_org = _loss_fn_org(L_o, y_org, device) 
                    loss = (cell_loss_coeff*loss_cell + org_loss_coeff*loss_org) / accumulation_steps

            scaler.scale(loss).backward()
            
            if (k // batch_size + 1) % accumulation_steps == 0 or (k + batch_size) >= nimg:
                scaler.unscale_(optimizer)
                # ---> GRADIENT CLIPPING AT 0.5 FOR STABILITY <---
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss = (loss.item() * accumulation_steps) * len(imgi)
            lavg += train_loss
            nsum += len(imgi)
            train_losses[iepoch] += train_loss
            
        train_losses[iepoch] /= nimg

        if auto_unfreeze and is_frozen and iepoch >= (n_epochs // 2):
            train_logger.info(f"\n>>> [AUTO-UNFREEZE] 50% Milestone Reached ({iepoch}/{n_epochs}). UNFREEZING WHOLE BACKBONE FOR FINE-TUNING!")
            for param in net.parameters(): 
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(net.parameters(), lr=LR[iepoch], weight_decay=weight_decay)
            is_frozen = False

        # --- BATCH VALIDATION EVALUATION (RUNS EVERY EPOCH) ---
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
                            loss = _loss_fn_seg(L_c, y_cell, device)
                        elif turnoff_cell_loss:
                            loss = _loss_fn_org(L_o, y_org, device) 
                        else:
                            loss_c = _loss_fn_seg(L_c, y_cell, device)
                            loss_o = _loss_fn_org(L_o, y_org, device) 
                            loss = cell_loss_coeff*loss_c + org_loss_coeff*loss_o
                    
                    lavgt += loss.item() * len(imgi)
            lavgt /= nimg_test
            test_losses[iepoch] = lavgt
                
        lavg /= nsum
        train_logger.info(f"Epoch {iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s")
        lavg, nsum = 0, 0

        # =====================================================================
        # FULL IMAGE POST-PROCESSING EVALUATION & VISUALIZATION (RUNS EVERY EPOCH)
        # =====================================================================
        if debug and test_data:
            temp_model_path = str(filename) + f"_eval_temp"
            net.save_model(temp_model_path)
            
            eval_model = model2.CellposeModel(gpu=True, custom_weights=temp_model_path, use_bfloat16=False, nchan=(6 if two_tail else 3))
            
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
                    
                    # --- ROW 1: CELLS ---
                    axes[0, 0].imshow(img_cell, cmap='gray')
                    if np.any(gt_cells[0]): axes[0, 0].contour(gt_cells[0] > 0, colors='lime', linewidths=1.2)
                    axes[0, 0].set_title(f"Cells - Actual GT ({n_cells_gt} masks)", fontsize=12)
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(img_cell, cmap='gray')
                    if np.any(pred_cells[0]): axes[0, 1].contour(pred_cells[0] > 0, colors='red', linewidths=1.2)
                    axes[0, 1].set_title(f"Cells - Predicted Reconstructed ({n_cells_pred} masks)", fontsize=12)
                    axes[0, 1].axis('off')
                    
                    # --- ROW 2: ORGANELLES ---
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

    if original_net_dtype != torch.float32: net.dtype = original_net_dtype

    if hf_repo_id and hf_token:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, exist_ok=True)
        api.upload_file(path_or_fileobj=str(filename), path_in_repo=f"models/{model_name}", repo_id=hf_repo_id, repo_type="model")

    return filename, train_losses, test_losses
