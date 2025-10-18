#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
same as train_eschernet_scannet.py but using config from config file
"""
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

# from unet_2d_condition import UNet2DConditionModel

LPIPS = lpips.LPIPS(net="alex", version="0.1")

if is_wandb_available():
    import wandb

# os.environ["WANDB_MODE"] = "offline"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.19.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


@torch.no_grad()
def log_validation(
    cfg,
    args,
    validation_dataloader,
    vae,
    image_encoder,
    feature_extractor,
    unet,
    accelerator,
    weight_dtype,
    split="val",
    use_ray=False,
    output_depth=False,
):
    logger.info("Running {} validation... ".format(split))

    scheduler_kwarg = {}
    if cfg.model.fix_scheduler_issue:
        assert cfg.model.prediction_type == "v_prediction"
        scheduler_kwarg = {
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing",
        }
    if cfg.model.prediction_type is not None:
        scheduler_kwarg["prediction_type"] = cfg.model.prediction_type
    scheduler = DDIMScheduler.from_pretrained(
        cfg.model.base_model,
        subfolder="scheduler",
        local_files_only=False,
        **scheduler_kwarg,
    )
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        cfg.model.base_model,
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=False,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    image_logs = []
    val_lpips = 0
    val_ssim = 0
    val_psnr = 0
    val_loss = 0
    val_num = 0
    T_out = cfg.model.T_out  # fix to be 1?
    for T_in_val in [
        cfg.model.T_in_val,
    ]:  # eval different number of given views
        for valid_step, batch in tqdm(enumerate(validation_dataloader)):
            if (
                cfg.validation.num_validation_batches is not None
                and valid_step >= cfg.validation.num_validation_batches
            ):
                break
            T_in = T_in_val
            gt_image = batch["image_target"].to(dtype=weight_dtype)
            input_image = batch["image_input"].to(dtype=weight_dtype)[:, :T_in]
            pose_in = batch["pose_in"].to(dtype=weight_dtype)[:, :T_in]  # BxTx4
            pose_out = batch["pose_out"].to(dtype=weight_dtype)  # BxTx4
            pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype)[:, :T_in]  # BxTx4
            pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype)  # BxTx4

            gt_image = einops.rearrange(gt_image, "b t c h w -> (b t) c h w", t=T_out)
            input_image = einops.rearrange(
                input_image, "b t c h w -> (b t) c h w", t=T_in
            )  # T_in

            kwargs = {}
            if use_ray:
                target_rays = [
                    v.to(dtype=weight_dtype)
                    for k, v in batch.items()
                    if "target_ray" in k
                ]
                kwargs["target_rays"] = target_rays

                cond_rays = [
                    v.to(dtype=weight_dtype)
                    for k, v in batch.items()
                    if "cond_ray" in k
                ]
                cond_rays = [
                    einops.rearrange(r[:, :T_in], "b t c h w -> (b t) c h w", t=T_in)
                    for r in cond_rays
                ]

                if len(cond_rays):
                    # NOTE: only support 1 cond_ray
                    kwargs["cond_rays"] = cond_rays[0]
            # prepare layout if any
            if "layout_cls" in batch:
                layout_dict = {}
                for k, v in batch.items():
                    if "layout" in k:
                        if v.dtype != torch.long:
                            v = v.to(dtype=weight_dtype)
                        layout_dict[k] = v
                kwargs["layouts"] = layout_dict
                # layout_pos = batch["layout_pos"].to(dtype=weight_dtype)
                # kwargs["layouts"] = {
                #     "layout_pos": layout_pos,
                #     "layout_cls": batch["layout_cls"],
                # }
            # if "back_layout_cls" in batch:
            #     kwargs["layouts"]["back_layout_cls"] = batch["back_layout_cls"]
            #     kwargs["layouts"]["back_layout_pos"] = batch["back_layout_pos"].to(
            #         dtype=weight_dtype
            #     )

            # in_pos3d
            in_pos3d = batch.get("in_pos3d", None)
            if in_pos3d is not None:
                in_pos3d = in_pos3d.to(dtype=weight_dtype)
            depth_mask_cond = batch.get("depth_mask_cond", None)
            if depth_mask_cond is not None:
                depth_mask_cond = depth_mask_cond.to(dtype=weight_dtype)
                _,_,h,w,_ = in_pos3d.shape
                # interpolate depth_mask_cond to in_pos3d size
                # in_pos3d: torch.Size([8, t, 7, 7, 3]) mask: torch.Size([8, t, 224, 224])
                depth_mask_cond = einops.rearrange(depth_mask_cond, "b t h w -> (b t) 1 h w")
                resized_mask = F.interpolate(depth_mask_cond, size=(h, w), mode='nearest')  # Shape: (8, 1, 7, 7)
                resized_mask = einops.rearrange(resized_mask, "(b t) 1 h w -> b t h w 1", t=T_in)
                in_pos3d = in_pos3d * resized_mask  


            if cfg.model.fix_scheduler_issue:
                kwargs["guidance_rescale"] = 0.7

            images = []
            h, w = input_image.shape[2:]
            for _ in range(cfg.validation.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(
                        input_imgs=input_image,
                        prompt_imgs=input_image,
                        poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
                        height=h,
                        width=w,
                        T_in=T_in,
                        T_out=pose_out.shape[1],
                        guidance_scale=cfg.model.guidance_scale,
                        num_inference_steps=50,
                        generator=generator,
                        output_type="numpy",
                        output_depth=output_depth,
                        in_pos3d=in_pos3d,
                        depth_mask_cond=depth_mask_cond,
                        **kwargs,
                    ).images

                # rgb and depth pred are interleave in batch dimension
                pred_image = torch.from_numpy(image * 2.0 - 1.0).permute(0, 3, 1, 2)
                images.append(pred_image)

                # split rgb and depth for evaluation
                if output_depth:
                    image = einops.rearrange(image, "(b n) h w c -> n b h w c", n=2)
                    depth = image[1]
                    image = image[0]
                # eval image
                pred_np = (image * 255).astype(np.uint8)  # [0,1]
                gt_np = (gt_image / 2 + 0.5).clamp(0, 1)
                gt_np = (gt_np.cpu().permute(0, 2, 3, 1).float().numpy() * 255).astype(
                    np.uint8
                )
                # for 1 image
                # pixel loss
                loss = F.mse_loss(pred_image[0], gt_image[0].cpu()).item()
                # LPIPS
                lpips = LPIPS(
                    pred_image[0], gt_image[0].cpu()
                ).item()  # [-1, 1] torch tensor
                # SSIM
                ssim = calculate_ssim(pred_np[0], gt_np[0], channel_axis=2)
                # PSNR
                psnr = cv2.PSNR(gt_np[0], pred_np[0])

                val_loss += loss
                val_lpips += lpips
                val_ssim += ssim
                val_psnr += psnr

                val_num += 1

                # eval depth
                # TODO
                if output_depth:
                    pass

            image_logs.append(
                {
                    "gt_image": gt_image,
                    "pred_images": images,
                    "input_image": input_image,
                }
            )

        pixel_loss = val_loss / val_num
        pixel_lpips = val_lpips / val_num
        pixel_ssim = val_ssim / val_num
        pixel_psnr = val_psnr / val_num

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                # need to use table, wandb doesn't allow more than 108 images
                assert cfg.validation.num_validation_images == 2
                table = wandb.Table(columns=["Input", "GT", "Pred1", "Pred2"])

                for log_id, log in enumerate(image_logs):
                    formatted_images = [[], [], []]  # [[input], [gt], [pred]]
                    pred_images = log["pred_images"]  # pred
                    input_image = log["input_image"]  # input
                    gt_image = log["gt_image"]  # GT

                    formatted_images[0].append(
                        wandb.Image(input_image, caption="{}_input".format(log_id))
                    )
                    formatted_images[1].append(
                        wandb.Image(gt_image, caption="{}_gt".format(log_id))
                    )

                    for sample_id, pred_image in enumerate(pred_images):  # n_samples
                        pred_image = wandb.Image(
                            pred_image, caption="{}_pred_{}".format(log_id, sample_id)
                        )
                        formatted_images[2].append(pred_image)

                    table.add_data(
                        *formatted_images[0], *formatted_images[1], *formatted_images[2]
                    )
                # TODO: add depth metrics
                tracker.log(
                    {
                        split: table,  # formatted_images
                        "{}_T{}_pixel_loss".format(split, T_in_val): pixel_loss,
                        "{}_T{}_lpips".format(split, T_in_val): pixel_lpips,
                        "{}_T{}_ssim".format(split, T_in_val): pixel_ssim,
                        "{}_T{}_psnr".format(split, T_in_val): pixel_psnr,
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

    # del pipeline
    # torch.cuda.empty_cache()
    # after validation, set the pipeline back to training mode
    unet.train()
    vae.eval()
    image_encoder.train()

    return image_logs


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


# ConvNextV2_preprocess = transforms.Compose(
#     [
#         transforms.Resize(
#             (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
#         ),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )


def _encode_image(
    feature_extractor,
    image_encoder,
    image,
    depth_mask_cond,
    device,
    dtype,
    do_classifier_free_guidance,
    cond_rays=None,
):
    # [-1, 1] -> [0, 1]
    image = (image + 1.0) / 2.0
    # image = ConvNextV2_preprocess(image)
    image_embeddings = image_encoder(image, rays=cond_rays, depth_mask = depth_mask_cond)  # bt, 768, 12, 12

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings  # .detach() # !we need keep image encoder gradient


def pyramid_noise_like(x, timesteps, discount=0.9):
    """
    x: (b, c, w, h)
    timesteps: (b,) in [0,1]
    """
    b, c, w_ori, h_ori = x.shape
    u = torch.nn.Upsample(size=(w_ori, h_ori), mode="bilinear")
    noise = torch.randn_like(x)
    scale = 1.5
    for i in range(10):
        r = np.random.random() * scale + scale  # Rather than always going 2x,
        w, h = max(1, int(w_ori / (r**i))), max(1, int(h_ori / (r**i)))
        noise += u(torch.randn(b, c, w, h, device=x.device)) * (
            timesteps[..., None, None, None] * discount**i
        )
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def main(args, cfg):
    sys.path.append("/mnt/DATA/personal_projects/eschernet3/6DoF")
    logging_dir = Path(cfg.dir.output_dir, cfg.dir.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.dir.output_dir, logging_dir=logging_dir
    )

    train_cfg = cfg.training
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        mixed_precision=train_cfg.mixed_precision,
        log_with=cfg.log_with,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.dir.output_dir is not None:
            os.makedirs(cfg.dir.output_dir, exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token, private=True
        #     ).repo_id

    # Load scheduler and models
    model_cfg = cfg.model
    if args.weights is None:
        weights_path = model_cfg.base_model
    else:
        weights_path = args.weights

    scheduler_kwarg = {}
    if model_cfg.fix_scheduler_issue:
        assert model_cfg.prediction_type == "v_prediction"
        scheduler_kwarg = {
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing",
        }

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_cfg.base_model,
        subfolder="scheduler",
        revision=args.revision,
        local_files_only=False,
        **scheduler_kwarg,
    )
    if args.weights is None:
        image_encoder = CN_encoder.from_pretrained(
            "facebook/convnextv2-tiny-22k-224",
            use_ray=model_cfg.use_cond_ray,
            local_files_only=False,
            im_size=cfg.data.params.cond_image_size,
        )
    else:
        image_encoder = CN_encoder.from_pretrained(
            weights_path,
            subfolder="image_encoder",
            use_ray=model_cfg.use_cond_ray,
            local_files_only=False,
            im_size=cfg.data.params.cond_image_size,
        )
    feature_extractor = None
    vae = AutoencoderKL.from_pretrained(
        weights_path, subfolder="vae", revision=args.revision, local_files_only=False
    )
    kwargs = OmegaConf.to_container(cfg.model.params) if "params" in cfg.model else {}
    UNet2DConditionModel = get_obj_from_str(cfg.model.unet_cls)
    unet = UNet2DConditionModel.from_pretrained(
        weights_path,
        subfolder="unet",
        revision=args.revision,
        local_files_only=False,
        **kwargs,
    )

    T_in = model_cfg.T_in
    T_in_val = model_cfg.T_in_val
    T_out = model_cfg.T_out
    use_depths_mask = False

    vae.eval()
    vae.requires_grad_(False)

    image_encoder.train()
    image_encoder.requires_grad_(True)

    unet.train()
    unet.requires_grad_(True)

    # Create EMA for the unet.
    if model_cfg.use_ema:
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_slicing()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if train_cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if train_cfg.scale_lr:
        train_cfg.learning_rate = (
            train_cfg.learning_rate
            * train_cfg.gradient_accumulation_steps
            * cfg.data.params.batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if train_cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        [
            {"params": unet.parameters(), "lr": train_cfg.learning_rate},
            {"params": image_encoder.parameters(), "lr": train_cfg.learning_rate},
        ],
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.adam_weight_decay,
        eps=train_cfg.adam_epsilon,
    )

    # print model info, learnable parameters, non-learnable parameters, total parameters, model size, all in billion
    def print_model_info(model):
        print("=" * 20)
        # print model class name
        print("model name: ", type(model).__name__)
        print(
            "learnable parameters(M): ",
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
        print(
            "non-learnable parameters(M): ",
            sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6,
        )
        print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
        print(
            "model size(MB): ",
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
        )

    print_model_info(unet)
    print_model_info(vae)
    print_model_info(image_encoder)

    # Init Dataset
    data_module = instantiate_from_config(cfg.data)
    train_dataloader = data_module.train_dataloader()
    train_log_dataloader = data_module.val_dataloader()
    validation_dataloader = data_module.val_dataloader()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_cfg.gradient_accumulation_steps
    )
    if train_cfg.max_train_steps is None:
        train_cfg.max_train_steps = (
            train_cfg.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
        """Warmup the learning rate"""
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
        """Decay the learning rate"""
        lr = (init_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / max_epoch)
        ) + min_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Prepare everything with our `accelerator`.
    (
        unet,
        image_encoder,
        optimizer,
        train_dataloader,
        validation_dataloader,
        train_log_dataloader,
    ) = accelerator.prepare(
        unet,
        image_encoder,
        optimizer,
        train_dataloader,
        validation_dataloader,
        train_log_dataloader,
    )

    if model_cfg.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, image_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_cfg.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        train_cfg.max_train_steps = (
            train_cfg.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    train_cfg.num_train_epochs = math.ceil(
        train_cfg.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        run_name = cfg.dir.output_dir.split("logs_")[1]
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": run_name}},
        )

    # Train!
    total_batch_size = (
        cfg.data.params.batch_size
        * accelerator.num_processes
        * train_cfg.gradient_accumulation_steps
    )
    do_classifier_free_guidance = model_cfg.guidance_scale > 1.0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {train_cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.params.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {train_cfg.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {train_cfg.max_train_steps}")
    logger.info(f" do_classifier_free_guidance = {do_classifier_free_guidance}")
    logger.info(f" conditioning_dropout_prob = {model_cfg.conditioning_dropout_prob}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.dir.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.dir.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, train_cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, train_cfg.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, image_encoder):
                gt_image = batch["image_target"].to(dtype=weight_dtype)  # BxTx3xHxW
                gt_image = einops.rearrange(
                    gt_image, "b t c h w -> (b t) c h w", t=T_out
                )

                input_image = batch["image_input"].to(dtype=weight_dtype)  # Bx3xHxW
                input_image = einops.rearrange(
                    input_image, "b t c h w -> (b t) c h w", t=T_in
                )
                pose_in = batch["pose_in"].to(dtype=weight_dtype)  # BxTx4
                pose_out = batch["pose_out"].to(dtype=weight_dtype)  # BxTx4
                pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype)  # BxTx4
                pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype)  # BxTx4

                gt_latents = vae.encode(gt_image).latent_dist.sample().detach()
                gt_latents = (
                    gt_latents * vae.config.scaling_factor
                )  # follow zero123, only target image latent is scaled

                # concat depth_latent to gt_latents
                if "depth_target" in batch:
                    depth_target = batch["depth_target"].to(
                        dtype=weight_dtype
                    )  # BxTxHxW
                    depth_target = einops.repeat(
                        depth_target, "b t h w -> (b t) n h w", t=T_out, n=3
                    )
                    depth_latent = (
                        vae.encode(depth_target).latent_dist.sample().detach()
                    )
                    if "depth_mask_target" in batch:
                        depth_mask_target = batch["depth_mask_target"].to(
                            dtype=weight_dtype
                        )
                        # try to do interpolation:
                        h, w = depth_latent.shape[-2:]
                        depth_mask_target = einops.rearrange(
                            depth_mask_target, "b t h w -> (b t) 1 h w"
                        )
                        depth_mask_target = F.interpolate(
                            depth_mask_target, size=(h, w), mode="nearest"
                        )
                        # depth_mask_target = einops.rearrange(depth_mask_target, "b h w -> (b t) 1 h w")
                        
                    depth_latent = depth_latent * vae.config.scaling_factor
                    gt_latents = torch.cat([gt_latents, depth_latent], dim=1)

                # Sample a random timestep for each image
                bsz = gt_latents.shape[0] // T_out
                noise = torch.randn_like(gt_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=gt_latents.device,
                )
                timesteps = timesteps.long()
                timesteps = einops.repeat(timesteps, "b -> (b t)", t=T_out)

                # Sample noise that we'll add to the latents
                if train_cfg.noise_type == "pyramid":
                    noise = pyramid_noise_like(
                        gt_latents,
                        timesteps / noise_scheduler.config.num_train_timesteps,
                    )
                else:
                    noise = torch.randn_like(gt_latents)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    gt_latents.to(dtype=torch.float32),
                    noise.to(dtype=torch.float32),
                    timesteps,
                ).to(dtype=gt_latents.dtype)

                model_kwargs = {}
                if model_cfg.use_ray:
                    target_rays = [
                        v.to(dtype=weight_dtype)
                        for k, v in batch.items()
                        if "target_ray" in k
                    ]
                    target_rays = [
                        einops.rearrange(x, "b t c h w -> (b t) c h w", t=T_out)
                        for x in target_rays
                    ]
                    model_kwargs["target_rays"] = target_rays

                    cond_rays = [
                        v.to(dtype=weight_dtype)
                        for k, v in batch.items()
                        if "cond_ray" in k
                    ]
                    cond_rays = [
                        einops.rearrange(x, "b t c h w -> (b t) c h w", t=T_in)
                        for x in cond_rays
                    ]
                    # assume single stride of any
                    cond_rays = None if not len(cond_rays) else cond_rays[0]
                else:
                    cond_rays = None

                # prepare layout
                if "layout_cls" in batch:
                    layout_dict = {}
                    for k, v in batch.items():
                        if "layout" in k:
                            if v.dtype != torch.long:
                                v = v.to(dtype=weight_dtype)
                            layout_dict[k] = einops.rearrange(v, "b t ... -> (b t) ...", t=T_out)
                    model_kwargs["layouts"] = layout_dict

                depth_mask_cond = None
                if "depth_mask_cond" in batch:
                    depth_mask_cond = batch["depth_mask_cond"].to(dtype=weight_dtype)

                # in_pos3d
                in_pos3d = None
                if "in_pos3d" in batch:
                    in_pos3d = batch["in_pos3d"].to(dtype=weight_dtype)
                    # inspection code, delete later
                    if "depth_mask_cond" in batch:
                        _,_,h,w,_ = in_pos3d.shape
                        # interpolate depth_mask_cond to in_pos3d size
                        # in_pos3d: torch.Size([8, t, 7, 7, 3]) mask: torch.Size([8, t, 224, 224])
                        depth_mask_cond = einops.rearrange(depth_mask_cond, "b t h w -> (b t) 1 h w")
                        resized_mask = F.interpolate(depth_mask_cond, size=(h, w), mode='nearest')  # Shape: (8, 1, 7, 7)
                        resized_mask = einops.rearrange(resized_mask, "(b t) 1 h w -> b t h w 1", t=T_in)
                        in_pos3d = in_pos3d * resized_mask  
                    in_pos3d = einops.rearrange(in_pos3d, "b t h w c -> b (t h w) c")
                print("in_pos3d", in_pos3d.shape if in_pos3d is not None else None)

                if do_classifier_free_guidance:
                    # support classifier-free guidance, randomly drop out 5%
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    random_p = torch.rand(bsz, device=gt_latents.device)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * model_cfg.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1, 1)

                    img_prompt_embeds = _encode_image(
                        feature_extractor,
                        image_encoder,
                        input_image,
                        depth_mask_cond,
                        gt_latents.device,
                        gt_latents.dtype,
                        False,
                        cond_rays=cond_rays,
                    )

                    # Final text conditioning.
                    img_prompt_embeds = einops.rearrange(
                        img_prompt_embeds, "(b t) l c -> b t l c", t=T_in
                    )
                    null_conditioning = torch.zeros_like(img_prompt_embeds).detach()
                    img_prompt_embeds = torch.where(
                        prompt_mask, null_conditioning, img_prompt_embeds
                    )
                    img_prompt_embeds = einops.rearrange(
                        img_prompt_embeds, "b t l c -> (b t) l c", t=T_in
                    )
                    prompt_embeds = torch.cat([img_prompt_embeds], dim=-1)

                    if in_pos3d is not None:
                        in_pos3d[prompt_mask.view(-1)] = 0

                    # mask for layout
                    if "layouts" in model_kwargs and cfg.model.layout_cfg:
                        layout_mask = torch.logical_and(
                            (random_p > 2 * model_cfg.conditioning_dropout_prob),
                            (random_p < 4 * model_cfg.conditioning_dropout_prob),
                        )
                        layout_mask = einops.repeat(layout_mask, "b -> (b t)", t=T_out)

                        for v in model_kwargs["layouts"].values():
                            v[layout_mask] = 0
                        
                else:
                    # Get the image_with_pose embedding for conditioning
                    prompt_embeds = _encode_image(
                        feature_extractor,
                        image_encoder,
                        input_image,
                        depth_mask_cond,
                        gt_latents.device,
                        gt_latents.dtype,
                        False,
                        cond_rays=cond_rays,
                    )

                prompt_embeds = einops.rearrange(
                    prompt_embeds, "(b t) l c -> b (t l) c", t=T_in
                )

                # noisy_latents (b T_out)
                latent_model_input = torch.cat([noisy_latents], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,  # (bxT_in) l 768
                    pose=[
                        [pose_out, pose_out_inv],
                        [pose_in, pose_in_inv],
                    ],  # (bxT_in) 4, pose_out - self-attn, pose_in - cross-attn
                    in_pos3d=in_pos3d,
                    **model_kwargs,
                ).sample

                # Get the target for loss depending on the prediction type
                if model_cfg.prediction_type is not None:
                    noise_scheduler.register_to_config(
                        prediction_type=model_cfg.prediction_type
                    )

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # inspection code, delete later
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                # mask should be applied to d_loss only
                if use_depths_mask:
                    loss[:, 4:] = loss[:, 4:] * depth_mask_target
                loss_log = {}
                if loss.size(1) == 8:
                    loss_log = {
                        "rgb_loss": loss[:, :4].mean().detach().item(),
                        "d_loss": loss[:, 4:].mean().detach().item(),
                    }
                loss = (loss.mean([1, 2, 3])).mean()
                if torch.isnan(loss) and not train_cfg.skip_nan_loss:
                    raise ValueError("NaN loss during training")
                skip = False
                if train_cfg.skip_nan_loss:
                    all_losses = accelerator.gather(loss)
                    if torch.any(torch.isnan(all_losses)):
                        print("############## invalid loss")
                        skip = True

                if not skip:
                    accelerator.backward(loss)

                    if train_cfg.zero_nan_grad:
                        params_to_clip = itertools.chain(
                            unet.parameters(), image_encoder.parameters()
                        )
                        valid_gradients = True
                        for param in params_to_clip:
                            if param.grad is not None:
                                valid_gradients = not (
                                    torch.isnan(param.grad).any()
                                    or torch.isinf(param.grad).any()
                                )
                                if not valid_gradients:
                                    break

                        if not valid_gradients:
                            print("############## invalid gradient")
                            optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        params_to_clip = itertools.chain(
                            unet.parameters(), image_encoder.parameters()
                        )
                        accelerator.clip_grad_norm_(
                            params_to_clip, train_cfg.max_grad_norm
                        )
                    optimizer.step()
                    # cosine
                    if global_step <= train_cfg.lr_warmup_steps:
                        warmup_lr_schedule(
                            optimizer,
                            global_step,
                            train_cfg.lr_warmup_steps,
                            1e-5,
                            train_cfg.learning_rate,
                        )
                    else:
                        cosine_lr_schedule(
                            optimizer,
                            global_step,
                            train_cfg.max_train_steps,
                            train_cfg.learning_rate,
                            1e-5,
                        )
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if model_cfg.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % train_cfg.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if train_cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.dir.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= train_cfg.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints)
                                    - train_cfg.checkpoints_total_limit
                                    + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        cfg.dir.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            cfg.dir.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # save pipeline
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if train_cfg.checkpoints_total_limit is not None:
                            pipelines = os.listdir(cfg.dir.output_dir)
                            pipelines = [
                                d for d in pipelines if d.startswith("pipeline")
                            ]
                            pipelines = sorted(
                                pipelines, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new pipeline, we need to have at _most_ `checkpoints_total_limit - 1` pipeline
                            if len(pipelines) >= train_cfg.checkpoints_total_limit:
                                num_to_remove = (
                                    len(pipelines)
                                    - train_cfg.checkpoints_total_limit
                                    + 1
                                )
                                removing_pipelines = pipelines[0:num_to_remove]

                                logger.info(
                                    f"{len(pipelines)} pipelines already exist, removing {len(removing_pipelines)} pipelines"
                                )
                                logger.info(
                                    f"removing pipelines: {', '.join(removing_pipelines)}"
                                )

                                for removing_pipeline in removing_pipelines:
                                    removing_pipeline = os.path.join(
                                        cfg.dir.output_dir, removing_pipeline
                                    )
                                    shutil.rmtree(removing_pipeline)

                        if model_cfg.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
                            model_cfg.base_model,
                            vae=accelerator.unwrap_model(vae),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            feature_extractor=feature_extractor,
                            unet=accelerator.unwrap_model(unet),
                            scheduler=noise_scheduler,
                            safety_checker=None,
                            torch_dtype=torch.float32,
                            local_files_only=False,  ####################
                        )
                        pipeline_save_path = os.path.join(
                            cfg.dir.output_dir, f"pipeline-{global_step}"
                        )
                        pipeline.save_pretrained(pipeline_save_path)
                        # del pipeline

                        # if args.push_to_hub:
                        #     print("Pushing to the hub ", repo_id)
                        #     upload_folder(
                        #         repo_id=repo_id,
                        #         folder_path=pipeline_save_path,
                        #         commit_message=global_step,
                        #         ignore_patterns=["step_*", "epoch_*"],
                        #         run_as_future=True,
                        #     )

                        if model_cfg.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                    if (
                        validation_dataloader is not None
                        and global_step % cfg.validation.validation_steps == 0
                    ):
                        torch.cuda.empty_cache()
                        if model_cfg.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs = log_validation(
                            cfg,
                            args,
                            validation_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            accelerator,
                            weight_dtype,
                            "val",
                            use_ray=model_cfg.use_ray,
                            output_depth=model_cfg.output_depth,
                        )
                        if model_cfg.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        torch.cuda.empty_cache()
                    if train_log_dataloader is not None and (
                        global_step % cfg.validation.validation_steps == 0
                        or global_step == 1
                    ):
                        torch.cuda.empty_cache()
                        if model_cfg.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        train_image_logs = log_validation(
                            cfg,
                            args,
                            train_log_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            accelerator,
                            weight_dtype,
                            "train",
                            use_ray=model_cfg.use_ray,
                            output_depth=True,
                        )
                        if model_cfg.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        torch.cuda.empty_cache()

            loss_epoch += loss.detach().item()
            num_train_elems += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": optimizer.param_groups[0]["lr"],
                "loss_epoch": loss_epoch / num_train_elems,
                "epoch": epoch,
            }
            logs.update(loss_log)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= train_cfg.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if model_cfg.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
            model_cfg.base_model,
            vae=accelerator.unwrap_model(vae),
            image_encoder=accelerator.unwrap_model(image_encoder),
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            torch_dtype=torch.float32,
            local_files_only=False,
        )
        pipeline_save_path = os.path.join(cfg.dir.output_dir, f"pipeline-{global_step}")
        pipeline.save_pretrained(pipeline_save_path)

        # if args.push_to_hub:
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=pipeline_save_path,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")
    args, unknown = parse_args()

    configs = [OmegaConf.load(cfg) for cfg in args.cfg]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    main(args, config)
