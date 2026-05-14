"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import os, time
from pathlib import Path
import numpy as np
from tqdm import trange
import torch
from scipy.ndimage import gaussian_filter
import gc
import cv2

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

    """

    def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True):
        """
        Initialize the CellposeModel.
        """
        if diam_mean is not None:
            models_logger.warning(
                "diam_mean argument are not used in v4.0.1+. Ignoring this argument..."
            )
        if model_type is not None:
            models_logger.warning(
                "model_type argument is not used in v4.0.1+. Ignoring this argument..."
            )
        if nchan is not None:
            models_logger.warning("nchan argument is deprecated in v4.0.1+. Ignoring this argument")

        ### assign model device
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu

        if pretrained_model is None:
            raise ValueError("Must specify a pretrained model, training from scratch is not implemented")
        
        ### create neural network
        if pretrained_model and not os.path.exists(pretrained_model):
            # check if pretrained model is in the models directory
            model_strings = get_user_models()
            all_models = MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if pretrained_model in all_models:
                pretrained_model = os.path.join(MODEL_DIR, pretrained_model)
            else:
                pretrained_model = os.path.join(MODEL_DIR, "cpsam")
                models_logger.warning(
                    f"pretrained model {pretrained_model} not found, using default model"
                )

        self.pretrained_model = pretrained_model
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.net = Transformer(dtype=dtype).to(self.device)

        if os.path.exists(self.pretrained_model):
            models_logger.info(f">>>> loading model {self.pretrained_model}")
            self.net.load_model(self.pretrained_model, device=self.device)
        else:
            if os.path.split(self.pretrained_model)[-1] != 'cpsam':
                raise FileNotFoundError('model file not recognized')
            cache_CPSAM_model_path()
            self.net.load_model(self.pretrained_model, device=self.device)
        
        
    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, 
             min_size=15, max_size_fraction=0.4, niter=None, 
             augment=False, tile_overlap=0.1, bsize=256, 
             compute_masks=True, progress=None,
             active_head='cells'):
        """ segment list of images x, or 4D array - Z x 3 x Y x X """
        if isinstance(x, list) or x.squeeze().ndim == 5:
            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i], 
                    batch_size=batch_size,
                    channel_axis=channel_axis, 
                    z_axis=z_axis,
                    normalize=normalize, 
                    invert=invert,
                    diameter=diameter[i] if isinstance(diameter, list) or
                        isinstance(diameter, np.ndarray) else diameter, 
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
                    active_head=active_head)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        ############# actual eval code ############
        # reshape image
        x = transforms.convert_image(x, channel_axis=channel_axis,
                                        z_axis=z_axis, 
                                        do_3D=(do_3D or stitch_threshold > 0))
        
        # Add batch dimension if not present
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        nimg = x.shape[0]
        
        image_scaling = 1.0
        if diameter is not None and diameter > 0:
            image_scaling = 30. / diameter


        # normalize image
        normalize_params = normalize_default
        if isinstance(normalize, dict):
            normalize_params = {**normalize_params, **normalize}
        elif not isinstance(normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params["normalize"] = normalize
            normalize_params["invert"] = invert

        # pre-normalize if 3D stack for stitching or do_3D
        do_normalization = True if normalize_params["normalize"] else False
        if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
            normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
            x = transforms.normalize_img(x, **normalize_params)
            do_normalization = False # do not normalize again
        else:
            if normalize_params["norm3D"] and nimg > 1 and do_normalization:
                models_logger.warning(
                    "normalize_params['norm3D'] is True but do_3D is False and stitch_threshold=0, so setting to False"
                )
                normalize_params["norm3D"] = False
        if do_normalization:
            x = transforms.normalize_img(x, **normalize_params)

        # Set the active head on the decoder
        if hasattr(self.net, 'out'):
            self.net.out.active_head = active_head
            
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
                    models_logger.info(f"smoothing flows with ZYX sigma={flow3D_smooth}")
                    dP = gaussian_filter(dP, [0, *flow3D_smooth])
                else: 
                    models_logger.warning(f"Could not do flow smoothing with {flow3D_smooth} either because its len was not 3 or no items were > 0, skipping flow3D_smoothing")
                torch.cuda.empty_cache()
                gc.collect()

            if compute_masks:
                # use user niter if specified, otherwise scale niter (200) with diameter
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
                masks = np.zeros(0) #pass back zeros if not compute_masks
            
            masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()
            all_masks.append(masks)
            all_flows.append([plot.dx_to_circ(dP), dP, cellprob])
            all_styles.append(styles)

        if active_head == 'both':
            return np.stack(all_masks, axis=0), all_flows, all_styles
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
        nimg = shape[0]

        if do_3D:
            Lz, Ly, Lx = shape[:-1]
            if rescale != 1.0 or (anisotropy is not None and anisotropy != 1.0):
                models_logger.info(f"resizing 3D image with anisotropy={anisotropy}")
                anisotropy = 1.0 if anisotropy is None else anisotropy
                if rescale != 1.0:
                    x = transforms.resize_image(x, Ly=int(Ly*rescale),
                                                Lx=int(Lx*rescale))
                x = transforms.resize_image(x.transpose(1,0,2,3),
                                        Ly=int(Lz*anisotropy*rescale), 
                                        Lx=int(Lx*rescale)).transpose(1,0,2,3)
            yf_raw, styles = run_3D(self.net, x,
                                batch_size=batch_size, augment=augment,  
                                tile_overlap=tile_overlap, 
                                bsize=bsize
                                )
        else:
            yf_raw, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                                batch_size=batch_size,  
                                tile_overlap=tile_overlap, 
                                rsz=rescale if rescale !=1.0 else None)

        # Slice dual-head outputs back into individual tensors if needed
        yf_dict = {}
        if active_head == 'both':
            midpoint = yf_raw.shape[-1] // 2 
            yf_dict['cells'] = yf_raw[..., :midpoint]
            yf_dict['organelles'] = yf_raw[..., midpoint:]
        else:
            yf_dict[active_head] = yf_raw

        outputs = {}
        for head, yf in yf_dict.items():
            if resample:
                if do_3D:
                    if rescale != 1.0 or Lz != yf.shape[0]:
                        models_logger.info(f"resizing 3D flows and cellprob to original image size for head: {head}")
                        if rescale != 1.0:
                            yf = transforms.resize_image(yf, Ly=Ly, Lx=Lx)
                        if Lz != yf.shape[0]:
                            yf = transforms.resize_image(yf.transpose(1, 0, 2, 3), Ly=Lz, Lx=Lx).transpose(1, 0, 2, 3)
                else:
                    if rescale != 1.0:
                        yf = transforms.resize_image(yf, shape[1], shape[2])
            
            # Extract cellprob and dP
            if do_3D:
                cellprob = yf[..., -1]
                dP = yf[..., :-1].transpose((3, 0, 1, 2))
            else:
                cellprob = yf[..., -1]
                dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
            
            outputs[head] = (dP, cellprob, styles.squeeze() if isinstance(styles, np.ndarray) else styles)

        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info("network run in %2.2fs" % (net_time))

        return outputs
    
    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
        """ compute masks from flows and cell probability """
        changed_device_from = None
        if self.device.type == "mps" and do_3D:
            models_logger.warning("MPS does not support 3D post-processing, switching to CPU")
            self.device = torch.device("cpu")
            changed_device_from = "mps"
        Lz, Ly, Lx = shape[:3]
        tic = time.time()
        if do_3D:
            masks = dynamics.resize_and_compute_masks(
                dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold, do_3D=do_3D,
                min_size=min_size, max_size_fraction=max_size_fraction, 
                resize=shape[:3] if (np.array(dP.shape[-3:])!=np.array(shape[:3])).sum() 
                        else None,
                device=self.device)
        else:
            nimg = shape[0]
            Ly0, Lx0 = cellprob[0].shape 
            resize = None if Ly0==Ly and Lx0==Lx else [Ly, Lx]
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                            mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                # turn off min_size for 3D stitching
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
                models_logger.info(
                    f"stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks"
                )
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                masks = utils.fill_holes_and_remove_small_masks(
                    masks, min_size=min_size)
            elif nimg > 1:
                models_logger.warning(
                    "3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only"
                )

        flow_time = time.time() - tic
        if shape[0] > 1:
            models_logger.info("masks created in %2.2fs" % (flow_time))
        
        if changed_device_from is not None:
            models_logger.info("switching back to device %s" % self.device)
            self.device = torch.device(changed_device_from)
        return masks
