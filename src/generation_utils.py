import sys
from collections import deque

import numpy as np
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from CN_encoder import CN_encoder
from diffusers import AutoencoderKL, DDIMInverseScheduler, DDIMScheduler
from diffusers.single_file_loader.single_file_utils import convert_ldm_vae_checkpoint
from einops import rearrange, repeat
from skimage.metrics import structural_similarity as calculate_ssim
from torchvision import transforms
from tqdm import tqdm

# from train_eschernet_scannet import log_validation, parse_args
# from unet_2d_condition import UNet2DConditionModel

LPIPS = lpips.LPIPS(net="alex", version="0.1")
import argparse
import json
import zlib

import lmdb
import matplotlib.pyplot as plt
import networkx as nx
from cfg_util import get_obj_from_str, instantiate_from_config
from data_modules.data_utils import (
    Torch3DRenderer,
    collate_fn,
    get_absolute_depth,
    make_pipeline_input,
    make_pipeline_input_cross_rgbd,
    render_target_view,
    render_target_view_torch3d,
)
from data_modules.front3d import Front3DPose
from omegaconf import OmegaConf
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline


def preprocess_image(img):
    """
    return
        img in np.uint8 [0,255]
        img in torch.float32 [0,255]
    """
    img = (img + 1) / 2.0 * 255.0
    img = torch.clamp(img, min=0.0, max=255.0)
    np_img = img.cpu().numpy()
    np_img = np.round(np_img).astype(np.uint8)
    return img, np_img


class GraphSearch2:
    """
    breath first search
    """

    def __init__(self, graph, start_node, cond_hop=2, target_hop=1):
        assert start_node in graph.nodes
        self.graph = graph
        self.visited = set()
        self.unvisited = set(graph.nodes)
        self.candidates = deque([start_node])
        self.cond_hop = cond_hop
        self.target_hop = target_hop

    def next_node(self):
        while len(self.candidates):
            next_node = self.candidates.popleft()
            if next_node not in self.visited:
                return next_node
        while len(self.unvisited):
            # choose an unvisited node
            next_node = list(self.unvisited)[0]
            if next_node not in self.visited:
                return next_node
        return None

    def set_visited(self, node):
        # assert node in self.unvisited
        self.visited.add(node)
        self.unvisited.remove(node)
        neighbors = sorted(list(self.graph.neighbors(node)))
        for neighbor in neighbors:
            if neighbor not in self.visited:
                if neighbor not in self.candidates:
                    self.candidates.append(neighbor)

    def get_cond_target(self, node):
        """
        get condition node and target node based on hop distance from the given node
        """
        # print('searching for the neighbors of', node)
        n_hop = max(self.cond_hop, self.target_hop)
        neighbor_by_hop = self.get_neighbors(node, n_hop)
        # print('neighbor_by_hop', neighbor_by_hop)
        target = set.union(*neighbor_by_hop[: self.target_hop + 1]) - self.visited
        cond = set.union(*neighbor_by_hop[: self.cond_hop + 1]).intersection(
            self.visited
        )
        return sorted(list(cond)), sorted(list(target))

    def get_neighbors(self, node, hop):
        neighbor_by_hop = [{node}]
        cummulated_neighbors = {node}
        for _ in range(hop):
            neighbors = [set(self.graph.neighbors(n)) for n in neighbor_by_hop[-1]]
            if len(neighbors) == 0:
                print("no neighbors")
                break
            neighbors = set.union(*neighbors) - cummulated_neighbors
            neighbor_by_hop.append(neighbors)
            cummulated_neighbors = cummulated_neighbors.union(neighbors)
        return neighbor_by_hop
    
    def get_cond_target_with_max(self, node, max_target_node=None):
        """
        get condition node and target node based on hop distance from the given node
        """
        n_hop = max(self.cond_hop, self.target_hop)
        neighbor_by_hop = self.get_neighbors(node, n_hop)
        cond = set.union(*neighbor_by_hop[: self.cond_hop + 1]).intersection(
            self.visited
        )
        if max_target_node is not None:
            target = []
            hop = 0
            while len(target) < max_target_node and hop < self.target_hop + 1:
                new_nodes = sorted(list(neighbor_by_hop[hop] - self.visited))
                if len(new_nodes) + len(target) > max_target_node:
                    new_nodes = new_nodes[:max_target_node - len(target)]
                target.extend(new_nodes)
                hop += 1
        else:
            target = set.union(*neighbor_by_hop[: self.target_hop + 1]) - self.visited
            target = sorted(list(target))
        return sorted(list(cond)), target


def make_pipeline(
    cfg, ckpt_path, weight_dtype, device, revision=True, inverse_ddim=False, vae_ft=None,
    clip_sample_range=None
):
    image_encoder = CN_encoder.from_pretrained(
        ckpt_path,
        subfolder="image_encoder",
        use_ray=cfg.model.use_cond_ray,
        local_files_only=True,
    ).to(device)
    feature_extractor = None
    vae = AutoencoderKL.from_pretrained(
        ckpt_path, subfolder="vae", revision=revision, local_files_only=True
    )
    if vae_ft is not None:
        ft_weight = torch.load(vae_ft, map_location="cpu")["state_dict"]
        vae_cfg = dict(vae.config)
        ft_weight = convert_ldm_vae_checkpoint(ft_weight, vae_cfg)
        vae.load_state_dict(ft_weight)
    vae = vae.to(device)

    unet_kwargs = OmegaConf.to_container(cfg.model.params) if "params" in cfg.model else {}
    UNet2DConditionModel = get_obj_from_str(cfg.model.unet_cls)
    unet = UNet2DConditionModel.from_pretrained(
        ckpt_path,
        subfolder="unet",
        revision=revision,
        local_files_only=True,
        **unet_kwargs,
    ).to(device)
    
    scheduler_kwargs = {}
    if clip_sample_range is not None:
        scheduler_kwargs["clip_sample_range"] = clip_sample_range
        scheduler_kwargs["clip_sample"] = True
    if cfg.model.prediction_type is not None:
        scheduler_kwargs["prediction_type"] = cfg.model.prediction_type
    if cfg.model.fix_scheduler_issue:
        assert cfg.model.prediction_type == "v_prediction"
        scheduler_kwargs["rescale_betas_zero_snr"] = True
        scheduler_kwargs["timestep_spacing"] = "trailing"
    scheduler = DDIMScheduler.from_pretrained(ckpt_path, subfolder="scheduler", **scheduler_kwargs)
    if inverse_ddim:
        inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            ckpt_path, subfolder="scheduler", **scheduler_kwargs
        )
    else:
        inverse_scheduler = None
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        vae=vae.eval(),
        image_encoder=image_encoder.eval(),
        feature_extractor=feature_extractor,
        unet=unet.eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
        inverse_scheduler=inverse_scheduler,
    )

    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention()
    return pipeline


def generate_ddim_inversion(
    pipeline,
    batch,
    weight_dtype,
    generator,
    device,
    depth_transform,
    use_ray=True,
    cfg_scale=3.0,
    output_depth=False,
    torch_renderer=None,
    ignore_prompts=False,
    ddim_inversion=True,
    gaussian_clip=0.0,
    cross=False,
    guidance_rescale=0.0
):
    # T_in = batch["image_input"].size(1)
    # input_image = batch["image_input"].to(dtype=weight_dtype).to(device)
    # pose_in = batch["pose_in"].to(dtype=weight_dtype).to(device)  # BxTx4
    # pose_out = batch["pose_out"].to(dtype=weight_dtype).to(device)  # BxTx4
    # pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype).to(device)  # BxTx4
    # pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype).to(device)  # BxTx4

    # # make args for ddim inversion
    # if ddim_inversion:
    #     # NOTE assume batch size is 1
    #     assert batch["depth_input"].size(0) == 1
    #     input_depth = batch["depth_input"].to(device)
    #     warp_img, warp_depth = render_target_view_torch3d(
    #         rearrange((input_image + 1) * (255 / 2), "b t c h w -> (b t) h w c").to(
    #             torch.float32
    #         ),
    #         rearrange(input_depth, "b t h w -> (b t) h w").to(torch.float32),
    #         rearrange(pose_in.to(torch.float32), "b t c d -> (b t) c d"),
    #         rearrange(pose_out.to(torch.float32), "b t c d -> (b t) c d"),
    #         FOV=90.0,
    #         is_3dfront=True,
    #         torch_renderer=torch_renderer,
    #         batch_size=1,
    #     )
    #     warp_img = rearrange(warp_img, "t h w c -> 1 t c h w")
    #     warp_img = (warp_img.to(dtype=weight_dtype) / 255.0 - 0.5) * 2
    #     warp_depth = [depth_transform(depth) for depth in warp_depth]
    #     warp_depth = rearrange(warp_depth, "t h w -> 1 t h w").to(dtype=weight_dtype)
    # else:
    #     warp_img = None
    #     warp_depth = None
    # input_image = rearrange(input_image, "b t c h w -> (b t) c h w")  # T_in

    # kwargs = {}
    # # make args for ray condition
    # if use_ray:
    #     target_rays = [
    #         v.to(dtype=weight_dtype, device=device)
    #         for k, v in batch.items()
    #         if "target_ray" in k
    #     ]
    #     kwargs["target_rays"] = target_rays

    #     cond_rays = [
    #         v.to(dtype=weight_dtype, device=device)
    #         for k, v in batch.items()
    #         if "cond_ray" in k
    #     ]
    #     cond_rays = [rearrange(r, "b t c h w -> (b t) c h w") for r in cond_rays]

    #     if len(cond_rays):
    #         # NOTE: only support 1 cond_ray
    #         kwargs["cond_rays"] = cond_rays[0]

    # # make args for layout
    # if "layout_cls" in batch:
    #     layout_pos = batch["layout_pos"].to(dtype=weight_dtype, device=device)
    #     kwargs["layouts"] = {
    #         "layout_pos": layout_pos,
    #         "layout_cls": batch["layout_cls"].to(device=device),
    #     }
    # if "back_layout_cls" in batch:
    #     kwargs["layouts"]["back_layout_cls"] = batch["back_layout_cls"].to(
    #         device=device
    #     )
    #     kwargs["layouts"]["back_layout_pos"] = batch["back_layout_pos"].to(
    #         dtype=weight_dtype, device=device
    #     )

    # # make args for 3d position from input depth
    # in_pos3d = batch.get("in_pos3d", None)
    # if in_pos3d is not None:
    #     in_pos3d = in_pos3d.to(dtype=weight_dtype, device=device)

    # h, w = input_image.shape[2:]
    if not cross:
        pipepline_kwargs = make_pipeline_input(
            batch,
            device,
            weight_dtype,
            use_ray,
            output_depth,
            generator,
            cfg_scale,
            depth_transform,
            torch_renderer,
            ddim_inversion,
            gaussian_clip,
        )
    else:
        pipepline_kwargs = make_pipeline_input_cross_rgbd(
            batch,
            device,
            weight_dtype,
            use_ray,
            output_depth,
            generator,
            cfg_scale,
            depth_transform,
            torch_renderer,
            ddim_inversion,
            gaussian_clip,
        )
    warp_img = pipepline_kwargs["warp_img"]
    with torch.autocast("cuda"):
        image = pipeline(
            ignore_prompts=ignore_prompts,
            guidance_rescale=guidance_rescale,
            **pipepline_kwargs,
        ).images

        # t c h w
        pred_image = torch.from_numpy(image * 2.0 - 1.0).permute(0, 3, 1, 2).to(device)
    return pred_image, warp_img
