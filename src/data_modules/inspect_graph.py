import sys

sys.path.append("/work/vig/hieu/escher/6DoF")
sys.path.append("/mnt/DATA/personal_projects/eschernet3/6DoF")
import argparse
import itertools
import logging
import math
import os
import shutil
from pathlib import Path

import cv2
import diffusers
import einops
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from cfg_util import get_obj_from_str, instantiate_from_config
from CN_encoder import CN_encoder
from diffusers import DDIMScheduler  # UNet2DConditionModel,
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from skimage.metrics import structural_similarity as calculate_ssim
from tqdm.auto import tqdm

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a Zero123 training script."
    )
    parser.add_argument(
        "--cfg",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        default=True,
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        default=True,
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_zero123_hf",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args, unknown = parser.parse_known_args(input_args)
    else:
        args, unknown = parser.parse_known_args()

    return args, unknown

def main(args, cfg):
    sys.path.append("/mnt/DATA/personal_projects/eschernet3/6DoF")
    logging_dir = Path(cfg.dir.output_dir, cfg.dir.logging_dir)
    train_cfg = cfg.training
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)
    model_cfg = cfg.model
    T_in = model_cfg.T_in
    T_in_val = model_cfg.T_in_val
    T_out = model_cfg.T_out
    use_depths_mask = model_cfg.use_depths_mask
    # Init Dataset
    data_module = instantiate_from_config(cfg.data)
    # train_dataloader = data_module.train_dataloader()
    # train_log_dataloader = data_module.train_log_dataloader()
    validation_dataloader = data_module.val_dataloader()
    global_step = 0
    first_epoch = 0
    weight_dtype = torch.float32

    # start data curation:
    # train_dataloader.dataset.generate_curation()

    # for epoch in range(first_epoch, train_cfg.num_train_epochs):
    #     loss_epoch = 0.0
    #     num_train_elems = 0
    #     for step, batch in enumerate(train_dataloader):
    #         gt_image = batch["image_target"].to(dtype=weight_dtype)  # BxTx3xHxW
    #         gt_image = einops.rearrange(
    #             gt_image, "b t c h w -> (b t) c h w", t=T_out
    #         )
    #         input_image = batch["image_input"].to(dtype=weight_dtype)  # Bx3xHxW
    #         input_image = einops.rearrange(
    #             input_image, "b t c h w -> (b t) c h w", t=T_in
    #         )
    #         print('gt_image', gt_image.shape)
    #         print('input_image', input_image.shape)
    #         pose_in = batch["pose_in"].to(dtype=weight_dtype)  # BxTx4
    #         pose_out = batch["pose_out"].to(dtype=weight_dtype)  # BxTx4
    #         pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype)  # BxTx4
    #         pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype)  # BxTx4
    #         break
    #     break




if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")
    args, unknown = parse_args()

    configs = [OmegaConf.load(cfg) for cfg in args.cfg]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    main(args, config)
