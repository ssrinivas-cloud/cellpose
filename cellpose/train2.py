import time
import os
import numpy as np
import scipy.ndimage
from cellpose import io, utils, models, dynamics
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

def _loss_fn_seg(lbl, y, device):
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, -2:]
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2.
    loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    loss = loss + loss2
    return loss

def _get_batch(inds, data=None, labels_c=None, labels_o=None):
    imgs = []
    for i in inds:
        img = data[i].copy()
        
        # 1. SHAPE FIX: If the image is (Height, Width, Channels), swap to (Channels, Height, Width)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)
            
        # 2. NORMALIZATION FIX: Standardize pixel values so the network doesn't explode
        img = normalize_img(img, axis=0) 
        
        imgs.append(img)
        
    # Slice [1:] to remove the raw instance mask, keeping [FlowY, FlowX, Cellprob]
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
    
    # Diameter calculation (based purely on cell size, not organelles)
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
              batch_size=8, learning_rate=1e-5, n_epochs=100, weight_decay=0.1, 
              rescale=False, scale_range=0.5, bsize=256,
              model_name=None, class_weights=None, hf_repo_id=None, hf_token=None, 
              save_flows=False, visualize=False, debug=False, auto_unfreeze=False, **kwargs):
    
    device = net.device
    original_net_dtype = net.dtype 
    if net.dtype == torch.bfloat16:
        train_logger.info(">>> Converting bfloat16 network to float32 for training")
        net.dtype = torch.float32

    # Load and compute dual flows
    out = _process_train_test_paired(train_data, train_labels_c, train_labels_o, test_data, test_labels_c, test_labels_o, device=device)
    train_flows_c, train_flows_o, test_flows_c, test_flows_o, diam_train, diam_test = out
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)
    nimg = len(train_data)
    nimg_test = len(test_data) if test_data else 0

    # ==========================================
    # Cosine Annealing Learning Rate Schedule
    # ==========================================
    warmup_epochs = min(10, n_epochs // 10) 
    cosine_epochs = max(0, n_epochs - warmup_epochs)
    LR_warmup = np.linspace(0, learning_rate, warmup_epochs)
    min_lr = learning_rate * 0.01 
    if cosine_epochs > 0:
        steps = np.arange(cosine_epochs)
        LR_cosine = min_lr + 0.5 * (learning_rate - min_lr) * (1 + np.cos(np.pi * steps / cosine_epochs))
    else:
        LR_cosine = np.array([])
    LR = np.concatenate([LR_warmup, LR_cosine])
    
    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    train_logger.info(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")

    # ==========================================
    # Auto-Unfreeze Setup
    # ==========================================
    is_frozen = False
    if auto_unfreeze:
        is_frozen = True
        train_logger.info("\n>>> [AUTO-UNFREEZE] Enabled! Phase 1: FROZEN BACKBONE (Training Heads Only).")
        for name, param in net.named_parameters():
            if 'out' in name: 
                param.requires_grad = True
            else: 
                param.requires_grad = False
        trainable_params = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and net.dtype in [torch.float16, torch.bfloat16]))
    accumulation_steps = max(1, 8 // batch_size)

    # Force dual head mode permanently for this run
    if hasattr(net, 'out') and hasattr(net.out, 'active_head'):
        net.out.active_head = 'both'

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() / "models"
    save_path.mkdir(exist_ok=True)
    filename = save_path / model_name
    
    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    best_loss, patience_counter, plateau_patience = float('inf'), 0, 5
    
    for iepoch in range(n_epochs):
        rperm = np.random.permutation(nimg)
        for param_group in optimizer.param_groups: 
            param_group["lr"] = LR[iepoch]
        
        net.train()
        optimizer.zero_grad()

        for k in range(0, nimg, batch_size):
            kend = min(k + batch_size, nimg)
            inds = rperm[k:kend]
            
            imgs, lbls_c, lbls_o = _get_batch(inds, data=train_data, labels_c=train_flows_c, labels_o=train_flows_o)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(len(diams), "float32")
            
            # MAGIC TRICK: Stack Cell and Organelle masks into one array to augment them perfectly in sync
            lbls_stacked = [np.concatenate((lbls_c[i], lbls_o[i]), axis=0) for i in range(len(inds))]
            
            imgi, lbl_aug = random_rotate_and_resize(imgs, Y=lbls_stacked, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize))[:2]
            
            # Split them back apart after random rotation and cropping
            lbl_c_aug = lbl_aug[:, :3, :, :]
            lbl_o_aug = lbl_aug[:, 3:, :, :]
                                                 
            X = torch.from_numpy(imgi).to(device)
            L_c = torch.from_numpy(lbl_c_aug).to(device)
            L_o = torch.from_numpy(lbl_o_aug).to(device)

            loss = torch.tensor(0.0, device=device)

            with torch.autocast(device_type=device.type, dtype=net.dtype):
                # Predict BOTH simultaneously
                y_cell, y_org = net(X)
                
                # ==========================================
                # DEBUG LOGGER
                # ==========================================
                if debug and k == 0:
                    train_logger.info(f"\n[DEBUG] --- EPOCH {iepoch} BATCH 0 ---")
                    train_logger.info(f"[DEBUG] Input Data (X) -> Shape: {X.shape}, Min: {X.min().item():.2f}, Max: {X.max().item():.2f}")
                    pred_cell_blobs = scipy.ndimage.label((y_cell[0, -1] > 0.0).detach().cpu().numpy())[1]
                    pred_org_blobs = scipy.ndimage.label((y_org[0, -1] > 0.0).detach().cpu().numpy())[1]
                    train_logger.info(f"[DEBUG] Dual-Head Prediction -> Cells: {pred_cell_blobs} | Orgs: {pred_org_blobs}")
                    train_logger.info(f"[DEBUG] ---------------------------\n")

                # Accumulate both losses
                loss_cell = _loss_fn_seg(L_c, y_cell, device)
                loss_org = _loss_fn_seg(L_o, y_org, device)
                loss = (loss_cell + loss_org) / accumulation_steps

            scaler.scale(loss).backward()
            
            if (k // batch_size + 1) % accumulation_steps == 0 or (k + batch_size) >= nimg:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss = (loss.item() * accumulation_steps) * len(imgi)
            lavg += train_loss
            nsum += len(imgi)
            train_losses[iepoch] += train_loss
            
        train_losses[iepoch] /= nimg

        # ==========================================
        # AUTO-UNFREEZE LOGIC
        # ==========================================
        if is_frozen and iepoch >= (n_epochs // 2):
            if train_losses[iepoch] < best_loss - 1e-4:
                best_loss, patience_counter = train_losses[iepoch], 0
            else: 
                patience_counter += 1
                
            if patience_counter >= plateau_patience:
                train_logger.info(f"\n>>> [AUTO-UNFREEZE] Plateau detected. Phase 2: UNFREEZING BACKBONE!")
                for name, param in net.named_parameters(): 
                    param.requires_grad = True
                LR[iepoch:] *= 0.1 # Drop LR by 90% to protect backbone
                optimizer = torch.optim.AdamW(net.parameters(), lr=LR[iepoch], weight_decay=weight_decay)
                is_frozen = False

        # ==========================================
        # VALIDATION EVALUATION
        # ==========================================
        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            if test_data:
                rperm_test = np.random.permutation(nimg_test)
                for ibatch in range(0, nimg_test, batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm_test[ibatch:ibatch + batch_size]
                        imgs, lbls_c, lbls_o = _get_batch(inds, data=test_data, labels_c=test_flows_c, labels_o=test_flows_o)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(len(diams), "float32")
                        
                        lbls_stacked = [np.concatenate((lbls_c[i], lbls_o[i]), axis=0) for i in range(len(inds))]
                        imgi, lbl_aug = random_rotate_and_resize(imgs, Y=lbls_stacked, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize))[:2]
                        lbl_c_aug, lbl_o_aug = lbl_aug[:, :3, :, :], lbl_aug[:, 3:, :, :]
                            
                        X = torch.from_numpy(imgi).to(device)
                        L_c = torch.from_numpy(lbl_c_aug).to(device)
                        L_o = torch.from_numpy(lbl_o_aug).to(device)

                        with torch.autocast(device_type=device.type, dtype=net.dtype):
                            y_cell, y_org = net(X)
                            loss_c = _loss_fn_seg(L_c, y_cell, device)
                            loss_o = _loss_fn_seg(L_o, y_org, device)
                            loss = loss_c + loss_o
                        
                        lavgt += loss.item() * len(imgi)
                lavgt /= nimg_test
                test_losses[iepoch] = lavgt
                
        lavg /= nsum
        train_logger.info(f"Epoch {iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s")
        lavg, nsum = 0, 0

        # ==========================================
        # VISUALIZATION LOGIC
        # ==========================================
        if iepoch % 10 == 0 and iepoch > 0 and test_data:
            train_logger.info(f">>> Running true multi-channel evaluation pipeline...")
            temp_model_path = str(filename) + f"_eval_temp"
            net.save_model(temp_model_path)
            eval_model = models.CellposeModel(gpu=True, custom_weights=temp_model_path)
            
            masks_both, _, _ = eval_model.eval(test_data, batch_size=2, channels=[0,0], cellprob_threshold=0.0, rescale=1.0, active_head='both')
            
            pred_cells = [m[0] for m in masks_both]
            pred_orgs = [m[1] for m in masks_both]
            gt_cells = [t[0] for t in test_flows_c]
            gt_orgs = [t[0] for t in test_flows_o]
            
            def calc_metrics(gt_masks, pred_masks):
                tp = fp = fn = 0
                for gt, pred in zip(gt_masks, pred_masks):
                    gt_bin, pred_bin = gt > 0, pred > 0
                    tp += np.logical_and(gt_bin, pred_bin).sum()
                    fp += np.logical_and(~gt_bin, pred_bin).sum()  
                    fn += np.logical_and(gt_bin, ~pred_bin).sum()  
                return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            iou_c = calc_metrics(gt_cells, pred_cells)
            iou_o = calc_metrics(gt_orgs, pred_orgs)
            train_logger.info(f"--- CELLS IOU: {iou_c:.4f} | ORGANELLES IOU: {iou_o:.4f} ---")

            if visualize:
                try:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
                    
                    # Extract the original Brightfield and Fluorescent channels to show
                    img_disp = test_data[0][..., :3].copy()
                    if img_disp.shape[0] == 3: img_disp = img_disp.transpose(1, 2, 0)
                    if img_disp.max() > 1.0: img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
                    
                    # Show Cell channel (Red/0) as grayscale
                    img_cell = img_disp[..., 0] 
                    axes[0, 0].imshow(img_cell, cmap='gray')
                    axes[0, 0].contour(gt_cells[0]>0, colors='lime', linewidths=0.5); axes[0, 0].set_title("Cells - Actual GT")
                    axes[0, 1].imshow(img_cell, cmap='gray')
                    axes[0, 1].contour(pred_cells[0]>0, colors='red', linewidths=0.5); axes[0, 1].set_title("Cells - Predicted")
                    
                    # Show Organelle channel (Green/1) as grayscale
                    img_org = img_disp[..., 1]
                    axes[1, 0].imshow(img_org, cmap='gray')
                    axes[1, 0].contour(gt_orgs[0]>0, colors='lime', linewidths=0.5); axes[1, 0].set_title("Organelles - Actual GT")
                    axes[1, 1].imshow(img_org, cmap='gray')
                    axes[1, 1].contour(pred_orgs[0]>0, colors='red', linewidths=0.5); axes[1, 1].set_title("Organelles - Predicted")
                    
                    plt.tight_layout()
                    vis_file = save_path / f"epoch_{iepoch:04d}_eval_vis.png"
                    plt.savefig(vis_file)
                    plt.close(fig)
                except Exception as e: 
                    train_logger.warning(f"Visualization failed: {e}")
                    
            del eval_model
            torch.cuda.empty_cache()

    if original_net_dtype != torch.float32:
        train_logger.info(f">>> converting network back to {original_net_dtype} after training")
        net.dtype = original_net_dtype

    if hf_repo_id and hf_token:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, exist_ok=True)
        api.upload_file(path_or_fileobj=str(filename), path_in_repo=f"models/{model_name}", repo_id=hf_repo_id, repo_type="model")

    return filename, train_losses, test_losses
