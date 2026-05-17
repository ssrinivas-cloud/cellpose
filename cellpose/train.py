import time
import os
import numpy as np
from cellpose import io, utils, models, dynamics
from cellpose.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
from tqdm import trange
import logging

from huggingface_hub import HfApi

train_logger = logging.getLogger(__name__)

def _loss_fn_class(lbl, y, class_weights=None):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        
    Returns:
        torch.Tensor: Loss value.

    """

    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    loss3 = criterion3(y[:, :-3], lbl[:, 0].long())
    
    return loss3

def _loss_fn_seg(lbl, y, device):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, -2:]
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2.
    loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    loss = loss + loss2
    return loss

def _reshape_norm(data, channel_axis=None, normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data, with channels axis first or last.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if (np.array([td.ndim!=3 for td in data]).sum() > 0 or
        np.array([td.shape[0]!=3 for td in data]).sum() > 0):
        data_new = []
        for td in data:
            if td.ndim == 3:
                channel_axis0 = channel_axis if channel_axis is not None else np.array(td.shape).argmin()
                # put channel axis first 
                td = np.moveaxis(td, channel_axis0, 0)
                td = td[:3] # keep at most 3 channels
            if td.ndim == 2 or (td.ndim == 3 and td.shape[0] == 1):
                td = np.stack((td, 0*td, 0*td), axis=0)
            elif td.ndim == 3 and td.shape[0] < 3:
                td = np.concatenate((td, 0*td[:1]), axis=0)
            data_new.append(td)
        data = data_new
    if normalize_params["normalize"]:
        data = [
            normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    return data

def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               normalize_params={"normalize": False}, tasks=None):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).
        tasks (list or None): List of task tags (0 for cell, 1 for organelle).

    Returns:
        tuple: A tuple containing lists: the batch of images, labels, and tasks.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i][1:] for i in inds]
        
    # Grab the task identifiers for this specific batch
    batch_tasks = [tasks[i] for i in inds] if tasks is not None else [0] * len(inds)
    
    return imgs, lbls, batch_tasks

def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}):
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = convert_image(td, channels=channels, channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, normalize_params={"normalize": False}, 
                        channel_axis=None, device=None):
    """
    Process train and test data.
    """
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None
    
    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files),
                                          files=train_files, device=device)
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files),
                                              files=test_files, device=device)

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if normalize_params["normalize"]:
        train_logger.info(f">>> normalizing {normalize_params}")
    if train_data is not None:
        train_data = _reshape_norm(train_data, channel_axis=channel_axis, 
                                   normalize_params=normalize_params)
        normed = True
    if test_data is not None:
        test_data = _reshape_norm(test_data, channel_axis=channel_axis,
                                  normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, train_tasks=None, 
              test_data=None, test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, test_tasks=None, channel_axis=None,
              load_files=True, batch_size=1, learning_rate=1e-5, SGD=False,
              n_epochs=100, weight_decay=0.1, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=False, scale_range=None, bsize=256,
              min_train_masks=5, model_name=None, class_weights=None,
              organelles=True, hf_repo_id=None, hf_token=None, save_flows=False, load_flows_dir=None, visualize=False):
    
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    original_net_dtype = net.dtype 
    if net.dtype == torch.bfloat16:
        train_logger.info(">>> converting bfloat16 network to float32 for training")
        net.dtype = torch.float32

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    # --- LOAD PRECOMPUTED FLOWS ---
    if load_flows_dir is not None and os.path.exists(load_flows_dir):
        flows_dir = Path(load_flows_dir)
        train_logger.info(f">>> Loading precomputed flows from {flows_dir} to skip computation...")
        try:
            if train_data is not None:
                train_labels = [np.load(flows_dir / f"train_flow_{i}.npy") for i in range(len(train_data))]
            if test_data is not None:
                test_labels = [np.load(flows_dir / f"test_flow_{i}.npy") for i in range(len(test_data))]
            train_logger.info(">>> Successfully loaded precomputed flows!")
        except Exception as e:
            train_logger.error(f">>> Failed to load precomputed flows: {e}")

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channel_axis=channel_axis,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
     
    # --- SAVE AND UPLOAD COMPUTED FLOWS ---
    if save_flows and load_flows_dir is None:
        save_path_obj = Path.cwd() if save_path is None else Path(save_path)
        flows_dir = save_path_obj / "computed_flows"
        flows_dir.mkdir(parents=True, exist_ok=True)
        
        train_logger.info(f">>> Saving computed flows locally to {flows_dir}")
        if train_labels is not None:
            for i, flow in enumerate(train_labels):
                np.save(flows_dir / f"train_flow_{i}.npy", flow)
        if test_labels is not None:
            for i, flow in enumerate(test_labels):
                np.save(flows_dir / f"test_flow_{i}.npy", flow)
                
        if hf_repo_id and hf_token:
            train_logger.info(f">>> Uploading flows to Hugging Face Hub: {hf_repo_id}")
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=hf_repo_id, exist_ok=True)
                api.upload_folder(
                    folder_path=str(flows_dir),
                    path_in_repo="computed_flows",
                    repo_id=hf_repo_id,
                    repo_type="model"
                )
                train_logger.info(">>> Flows upload successful!")
            except Exception as e:
                train_logger.error(f">>> Failed to upload flows to Hugging Face: {e}")

    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # ==========================================
    # BETTER LR SCHEDULE: Cosine Annealing 
    # ==========================================
    warmup_epochs = min(10, n_epochs // 10) # Warmup for 10 epochs or 10% of total
    cosine_epochs = max(0, n_epochs - warmup_epochs)
    
    # 1. Warmup (Linear increase from 0 to learning_rate)
    LR_warmup = np.linspace(0, learning_rate, warmup_epochs)
    
    # 2. Cosine Decay (Smooth curve from learning_rate down to 1% of learning_rate)
    min_lr = learning_rate * 0.01 
    if cosine_epochs > 0:
        steps = np.arange(cosine_epochs)
        LR_cosine = min_lr + 0.5 * (learning_rate - min_lr) * (1 + np.cos(np.pi * steps / cosine_epochs))
    else:
        LR_cosine = np.array([])
        
    LR = np.concatenate([LR_warmup, LR_cosine])
    # ==========================================

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize GradScaler for mixed precision stability (Fixed deprecation warning)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and net.dtype in [torch.float16, torch.bfloat16]))

    # Set Accumulation Steps (Simulates a larger batch size)
    accumulation_steps = 8

    # DYNAMICALLY SET MULTI-HEAD MODE BEFORE LOOP
    if hasattr(net, 'out') and hasattr(net.out, 'active_head'):
        net.out.active_head = 'both' if organelles else 'cells'

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            # choose random images for epoch with probability train_probs
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            # otherwise use all images
            rperm = np.random.permutation(np.arange(0, nimg))
            
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch] # set learning rate
            
        net.train()
        if hasattr(net, 'out') and hasattr(net.out, 'active_head'):
            net.out.active_head = 'both' if organelles else 'cells'

        # Zero gradients BEFORE the inner loop begins
        optimizer.zero_grad()

        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            
            # Fetch batch tasks alongside images and labels
            imgs, lbls, batch_tasks = _get_batch(inds, data=train_data, labels=train_labels,
                                                 files=train_files, labels_files=train_labels_files,
                                                 tasks=train_tasks, **kwargs)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
                
            # augmentations
            imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                 scale_range=scale_range,
                                                 xy=(bsize, bsize))[:2]
                                                 
            # network and loss optimization
            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)
            
            loss = torch.tensor(0.0, device=device)

            with torch.autocast(device_type=device.type, dtype=net.dtype):
                # PROPERLY UNPACK OUTPUTS AND STYLE
                outputs, style = net(X)
                
                # BRANCH BASED ON `organelles` FLAG
                if organelles:
                    y_cell, y_org = outputs
                    
                    # Convert tasks to tensor for boolean masking
                    batch_tasks_tensor = torch.tensor(batch_tasks, device=device)
                    cell_mask = (batch_tasks_tensor == 0)
                    org_mask = (batch_tasks_tensor == 1)

                    # Route cell loss
                    if cell_mask.any():
                        loss_cell = _loss_fn_seg(lbl[cell_mask], y_cell[cell_mask], device)
                        if y_cell.shape[1] > 3:
                            loss_cell += _loss_fn_class(lbl[cell_mask], y_cell[cell_mask], class_weights=class_weights)
                        loss += loss_cell
                        
                    # Route organelle loss
                    if org_mask.any():
                        loss_org = _loss_fn_seg(lbl[org_mask], y_org[org_mask], device)
                        if y_org.shape[1] > 3:
                            loss_org += _loss_fn_class(lbl[org_mask], y_org[org_mask], class_weights=class_weights)
                        loss += loss_org
                        
                else:
                    # STANDARD SINGLE-HEAD LOGIC
                    y_cell = outputs
                    loss_cell = _loss_fn_seg(lbl, y_cell, device)
                    if y_cell.shape[1] > 3:
                        loss_cell += _loss_fn_class(lbl, y_cell, class_weights=class_weights)
                    loss += loss_cell

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            # Use Scaler for backward pass
            scaler.scale(loss).backward()
            
            # Optimizer Step with Accumulation and Gradient Clipping
            if (k // batch_size + 1) % accumulation_steps == 0 or (k + batch_size) >= nimg_per_epoch:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent massive spikes
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                # Step and update scaler
                scaler.step(optimizer)
                scaler.update()
                
                # Zero out gradients for the next accumulation cycle
                optimizer.zero_grad()
            
            # Correct the reported train loss to reflect the un-scaled magnitude
            train_loss = (loss.item() * accumulation_steps) * len(imgi)

            # keep track of average training loss across epochs
            lavg += train_loss
            nsum += len(imgi)
            # per epoch training loss
            train_losses[iepoch] += train_loss
            
        train_losses[iepoch] /= nimg_per_epoch

        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test),
                                             size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                    
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        if hasattr(net, 'out') and hasattr(net.out, 'active_head'):
                            net.out.active_head = 'both' if organelles else 'cells'
                            
                        inds = rperm[ibatch:ibatch + batch_size]
                        
                        imgs, lbls, batch_tasks = _get_batch(inds, data=test_data,
                                                             labels=test_labels, files=test_files,
                                                             labels_files=test_labels_files,
                                                             tasks=test_tasks, **kwargs)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(
                            len(diams), "float32")
                        imgi, lbl = random_rotate_and_resize(
                            imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                            xy=(bsize, bsize))[:2]
                            
                        X = torch.from_numpy(imgi).to(device)
                        lbl = torch.from_numpy(lbl).to(device)
                        
                        loss = torch.tensor(0.0, device=device)

                        with torch.autocast(device_type=device.type, dtype=net.dtype):
                            outputs, style = net(X)
                            
                            if organelles:
                                y_cell, y_org = outputs
                                
                                batch_tasks_tensor = torch.tensor(batch_tasks, device=device)
                                cell_mask = (batch_tasks_tensor == 0)
                                org_mask = (batch_tasks_tensor == 1)

                                if cell_mask.any():
                                    loss_cell = _loss_fn_seg(lbl[cell_mask], y_cell[cell_mask], device)
                                    if y_cell.shape[1] > 3:
                                        loss_cell += _loss_fn_class(lbl[cell_mask], y_cell[cell_mask], class_weights=class_weights)
                                    loss += loss_cell
                                    
                                if org_mask.any():
                                    loss_org = _loss_fn_seg(lbl[org_mask], y_org[org_mask], device)
                                    if y_org.shape[1] > 3:
                                        loss_org += _loss_fn_class(lbl[org_mask], y_org[org_mask], class_weights=class_weights)
                                    loss += loss_org
                            else:
                                y_cell = outputs
                                loss_cell = _loss_fn_seg(lbl, y_cell, device)
                                if y_cell.shape[1] > 3:
                                    loss_cell += _loss_fn_class(lbl, y_cell, class_weights=class_weights)
                                loss += loss_cell
                        
                        test_loss = loss.item()
                        test_loss *= len(imgi)
                        lavgt += test_loss
                        
                lavgt /= len(rperm)
                test_losses[iepoch] = lavgt
                
        # Calculate and log per-epoch stats
        lavg /= nsum
        train_logger.info(
            f"Epoch {iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
        )
        lavg, nsum = 0, 0

        # --- EVALUATION AND VISUALIZATION EVERY 10 EPOCHS ---
        if iepoch % 10 == 0 and iepoch > 0 and test_data is not None and organelles:
            train_logger.info(f">>> Running requested full evaluation pipeline for Epoch {iepoch}...")
            
            # 1. Save weights explicitly for the eval block to load
            temp_model_path = str(filename) + f"_eval_temp_epoch_{iepoch:04d}"
            net.save_model(temp_model_path)
            
            # 2. Load the model using custom_weights to inject the dual-head architecture
            eval_model = models.CellposeModel(gpu=True, custom_weights=temp_model_path)
            
            # 3. Run inference with active_head='both'
            train_logger.info(f">>> Running dual-head inference on {len(test_data)} test images...")
            masks_both, flows_both, styles_both = eval_model.eval(
                test_data, 
                batch_size=2, 
                channels=[0,0], 
                cellprob_threshold=0.0, 
                rescale=1.0,               # Change this if you used a specific rescale factor
                active_head='both'         # <--- The magic switch
            )
            
            # masks_both contains a pair of masks [Cell_Mask, Org_Mask] for EACH image
            # We unpack them into two separate lists across all images
            pred_cells = [m[0] for m in masks_both]
            pred_orgs = [m[1] for m in masks_both]
            
            # 4. Helper Function for Pixel-Wise Metrics
            def calculate_metrics(gt_masks, pred_masks):
                tp = fp = fn = 0
                for gt, pred in zip(gt_masks, pred_masks):
                    # Binarize masks for Pixel-wise logic
                    gt_bin = gt > 0
                    pred_bin = pred > 0
                    
                    tp += np.logical_and(gt_bin, pred_bin).sum()
                    fp += np.logical_and(~gt_bin, pred_bin).sum()  # gt == 0 and pred > 0
                    fn += np.logical_and(gt_bin, ~pred_bin).sum()  # gt > 0 and pred == 0

                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                return iou, f1
            
            # Extract dynamically using test_tasks (0 = Cells, 1 = Organelles)
            cell_indices = [i for i, t in enumerate(test_tasks) if t == 0]
            org_indices = [i for i, t in enumerate(test_tasks) if t == 1]
            
            # Slice out the cell ground truths and predictions
            gt_cells_only = [test_labels[i][0] for i in cell_indices] # Extract spatial mask
            pred_cells_only = [pred_cells[i] for i in cell_indices]

            # Slice out the organelle ground truths and predictions
            gt_orgs_only = [test_labels[i][0] for i in org_indices]   # Extract spatial mask
            pred_orgs_only = [pred_orgs[i] for i in org_indices]
            
            # Compute
            iou_cells, f1_cells = calculate_metrics(gt_cells_only, pred_cells_only)
            iou_orgs, f1_orgs = calculate_metrics(gt_orgs_only, pred_orgs_only)

            train_logger.info("\n--- FINAL EVALUATION SCORES ---")
            train_logger.info(f"CELLS      | IOU: {iou_cells:.4f} | F1: {f1_cells:.4f}")
            train_logger.info(f"ORGANELLES | IOU: {iou_orgs:.4f} | F1: {f1_orgs:.4f}")

            # 5. Visualization Block (2x2 Grid)
            if visualize:
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as mpatches
                    
                    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
                    
                    # Plot Row 0: Cells
                    if len(cell_indices) > 0:
                        c_idx = cell_indices[0] # Grab first available cell image
                        img_c = test_data[c_idx]
                        
                        # Format image array for matplotlib
                        img_disp_c = img_c.transpose(1, 2, 0) if img_c.shape[0] == 3 else img_c
                        if img_disp_c.max() > 1.0:
                            img_disp_c = (img_disp_c - img_disp_c.min()) / (img_disp_c.max() - img_disp_c.min())
                            
                        gt_mask_c = gt_cells_only[0]
                        pred_mask_c = pred_cells_only[0]
                        
                        # Top-Left: Cell GT
                        axes[0, 0].imshow(img_disp_c)
                        if np.any(gt_mask_c > 0):
                            axes[0, 0].contour(gt_mask_c, levels=np.unique(gt_mask_c), colors='lime', linewidths=1.0)
                        axes[0, 0].set_title(f"Epoch {iepoch} | Cells - Actual GT", fontsize=14, fontweight='bold')
                        axes[0, 0].axis('off')
                        
                        # Top-Right: Cell Prediction
                        axes[0, 1].imshow(img_disp_c)
                        if np.any(pred_mask_c > 0):
                            axes[0, 1].contour(pred_mask_c, levels=np.unique(pred_mask_c), colors='red', linewidths=1.0)
                        axes[0, 1].set_title(f"Epoch {iepoch} | Cells - Predicted", fontsize=14, fontweight='bold')
                        axes[0, 1].axis('off')
                        
                    # Plot Row 1: Organelles
                    if len(org_indices) > 0:
                        o_idx = org_indices[0] # Grab first available organelle image
                        img_o = test_data[o_idx]
                        
                        # Format image array for matplotlib
                        img_disp_o = img_o.transpose(1, 2, 0) if img_o.shape[0] == 3 else img_o
                        if img_disp_o.max() > 1.0:
                            img_disp_o = (img_disp_o - img_disp_o.min()) / (img_disp_o.max() - img_disp_o.min())
                            
                        gt_mask_o = gt_orgs_only[0]
                        pred_mask_o = pred_orgs_only[0]
                        
                        # Bottom-Left: Organelle GT
                        axes[1, 0].imshow(img_disp_o)
                        if np.any(gt_mask_o > 0):
                            axes[1, 0].contour(gt_mask_o, levels=np.unique(gt_mask_o), colors='lime', linewidths=1.0)
                        axes[1, 0].set_title(f"Epoch {iepoch} | Organelles - Actual GT", fontsize=14, fontweight='bold')
                        axes[1, 0].axis('off')
                        
                        # Bottom-Right: Organelle Prediction
                        axes[1, 1].imshow(img_disp_o)
                        if np.any(pred_mask_o > 0):
                            axes[1, 1].contour(pred_mask_o, levels=np.unique(pred_mask_o), colors='red', linewidths=1.0)
                        axes[1, 1].set_title(f"Epoch {iepoch} | Organelles - Predicted", fontsize=14, fontweight='bold')
                        axes[1, 1].axis('off')

                    plt.tight_layout()
                    
                    vis_save_path = save_path / "visualizations"
                    vis_save_path.mkdir(exist_ok=True)
                    vis_file = vis_save_path / f"epoch_{iepoch:04d}_eval_vis.png"
                    plt.savefig(vis_file, bbox_inches='tight')
                    plt.close(fig)
                    train_logger.info(f">>> Full evaluation visual report saved to {vis_file}")
                    
                except Exception as e:
                    train_logger.warning(f"Failed to generate full evaluation visualization: {e}")
                    
            # 6. Safety Cleanup - Free up GPU memory so the training loop can continue safely
            del eval_model
            torch.cuda.empty_cache()

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if save_each and iepoch != n_epochs - 1:  #separate files as model progresses
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)
    
    net.save_model(filename)
    
    if original_net_dtype != torch.float32:
        train_logger.info(f">>> converting network back to {original_net_dtype} after training")
        net.dtype = original_net_dtype

    # --- Hugging Face Upload Logic ---
    if hf_repo_id and hf_token:
        train_logger.info(f">>> Uploading model to Hugging Face Hub: {hf_repo_id}")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo_id, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(filename),
                path_in_repo=f"models/{model_name}",
                repo_id=hf_repo_id,
                repo_type="model"
            )
            train_logger.info(">>> Upload successful!")
        except Exception as e:
            train_logger.error(f">>> Failed to upload to Hugging Face: {e}")

    return filename, train_losses, test_losses
