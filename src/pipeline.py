import open3d as o3d
import numpy as np
import torch
from open3d.cuda.pybind.geometry import TriangleMesh
from typing import Union
import trimesh
import lmdb
import zlib
import os
import torch.nn as nn
import json
import cv2
from sklearn.neighbors import KDTree
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    Textures,
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.structures import Meshes
from einops import rearrange, repeat
from tqdm import tqdm
from infer_eval import psnr
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from data_modules.data_utils import (
    load_background_depth,
    Torch3DRenderer,
    render_target_view_torch3d,
    collate_fn,
    get_absolute_depth,
)
from collections import defaultdict
from accelerate import Accelerator
from generate_scene_relative_depth_no_cond import make_pipeline, make_dataset
from cfg_util import instantiate_from_config
from glob import glob
import argparse
from data_modules.mesh_renderer import TorchMeshRenderer
from reconstruct_utils import Open3DFuser, DB
from omegaconf import OmegaConf


def process_depth(depth):
    """
    args:
        depth torch.float32 in meter (h w)
    return
        a_depth in np.uint16 in mm (h w)
    """
    depth = depth * 1000
    assert depth.max() < 2**16
    depth = depth.cpu().numpy()
    depth = np.round(depth).astype(np.uint16)
    return depth


def process_image(img):
    """
    args:
        img torch.float32 in [0,1] (h w c)
    return
        img in np.uint8 [0,255]
    """
    img = img * 255.0
    img = torch.clamp(img, min=0.0, max=255.0)
    np_img = img.cpu().numpy()
    np_img = np.round(np_img).astype(np.uint8)
    return np_img


class Refinement:
    def __init__(
        self,
        mesh_path,
        img_db_path,
        layout_db_path,
        output_dir,
        meta=None,
        background_thr=0.65,
        img_size=256,
        device="cuda",
    ):
        if not os.path.exists(mesh_path):
            self._mesh = None
        else:
            self._mesh = TorchMeshRenderer.o3d_mesh_to_torch(
                o3d.io.read_triangle_mesh(mesh_path), device
            )
        self.mesh_path = mesh_path
        self.device = device
        self.output_dir = output_dir
        self.img_db_path = img_db_path
        self.layout_db_path = layout_db_path
        self.background_thr = background_thr
        self.img_size = img_size
        self._rgbs = {}  # np hwc [0,255]
        self._depths = {}  # np hw in meter
        self._r_depths = {}  # torch hw in meter
        self._r_rgbs = {}  # torch hwc [0,255]
        self._poses = {}  # np 4,4
        self._scores = {}
        self._meta_path = meta

        self._foreground_frames = None
        fg_frame_path = os.path.join(output_dir, "foreground_frames.json")
        if os.path.exists(fg_frame_path):
            self._foreground_frames = json.load(open(fg_frame_path))

        # portion of B when warp A to B
        self._correspondence = {}  # (A,B) -> score
        correspondence_path = os.path.join(output_dir, "correspondence_score.json")
        if os.path.exists(correspondence_path):
            self._correspondence = json.load(open(correspondence_path))

        self._low_quality_frames = None
        low_quality_path = os.path.join(output_dir, "low_quality_frames.json")
        if os.path.exists(low_quality_path):
            self._low_quality_frames = json.load(open(low_quality_path))

        self._kdtree = None
        self._img_db = lmdb.open(
            self.img_db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if layout_db_path:
            self._layput_db = lmdb.open(
                self.layout_db_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def _get_keys(self):
        assert self._meta_path is not None
        meta = json.load(open(self._meta_path))
        meta = list(meta.values())[0]
        frames = sum([x["frame_ids"]["target"] for x in meta], [])
        depth_methods = sum([x["depth_method"] for x in meta], [])
        frames = [x for x, y in zip(frames, depth_methods) if y == "layout"]
        self._keys = sorted(frames)
        return self._keys

    def get_full_keys(self):
        with self._img_db.begin() as txn:
            exist_keys = list(txn.cursor().iternext(values=False))
            exist_keys = [key.decode() for key in exist_keys]
            exist_keys = [key for key in exist_keys if key.endswith("_rgb")]
            exist_keys = [key.replace("_rgb", "") for key in exist_keys]
            exist_keys = sorted(exist_keys)
        return exist_keys

    def make_render_db(self, out_db_path):
        keys = self.get_full_keys()
        des_db = lmdb.open(out_db_path, map_size=int(1e12))
        for key in tqdm(keys):
            _, _, pose = self.get_rgbd_pose(key, return_rgb=False, return_depth=False)
            r_rgb, r_depth = self.get_render_rgbd(key)
            r_rgb = process_image(r_rgb)
            r_depth = process_depth(r_depth)
            save_rgbd_db(key, r_rgb, r_depth, pose, des_db)

    def get_rgbd_pose(self, key, return_pose=True, return_rgb=True, return_depth=True):
        """
        rgb: hwc np [0,255]
        depth hw np in meter
        pose c2w 4,4 np
        """
        if key in self._rgbs:
            return self._rgbs[key], self._depths[key], self._poses[key]

        pose, rgb, depth = None, None, None
        with self._img_db.begin() as txn:
            if return_pose:
                pose_key = key + "_pose"
                pose = txn.get(pose_key.encode("ascii"))
                pose = np.frombuffer(zlib.decompress(pose), dtype=np.float32)
                pose = pose.reshape(4, 4).copy()
                pose[3, 3] = 1
                self._poses[key] = pose

            if return_rgb:
                rgb_key = key + "_rgb"
                rgb = txn.get(rgb_key.encode("ascii"))
                rgb = np.frombuffer(rgb, dtype=np.uint8)
                rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
            if return_depth:
                depth_key = key + "_depth"
                depth = txn.get(depth_key.encode("ascii"))
                depth = zlib.decompress(depth)
                depth = np.frombuffer(depth, dtype=np.uint16)
                size = int(np.sqrt(len(depth)))
                depth = depth.reshape(size, size)
                depth = depth.astype(np.float32) * 0.001
        if return_rgb & return_depth & return_pose:
            self._rgbs[key] = rgb
            self._depths[key] = depth
            self._poses[key] = pose
        return rgb, depth, pose

    def get_render_rgbd(self, key):
        """
        rgb: hwc torch [0,255]
        depth hw torch in meter
        """
        _, _, pose = self.get_rgbd_pose(key, return_rgb=False, return_depth=False)
        r_rgb, r_depth = self.renderer.render(
            self.mesh, torch.tensor(pose).unsqueeze(0)
        )
        r_rgb = (r_rgb[0] * 255).cpu()
        r_depth = r_depth[0].cpu()
        self._r_rgbs[key] = r_rgb
        self._r_depths[key] = r_depth
        return r_rgb, r_depth

    @property
    def foreground_frames(self):
        """
        filter frames with high background (above a threshold)
        the foreground/background is determined by the layout condition
        """
        if self._foreground_frames is not None:
            return self._foreground_frames
        foreground_frames = []
        for key in self._keys:
            _, bg_mask = load_background_depth(self._layput_db, key, 1, "cpu")
            if bg_mask.float().mean() < self.background_thr:
                foreground_frames.append(key)
        self._foreground_frames = foreground_frames
        return foreground_frames

    def get_neighbors(self, k, distance=3.0):
        """
        get neighbors of a key within a distance
        """
        if self._kdtree is None:
            locations = [self._poses[key][:3, 3] for key in self._keys]
            locations = np.stack(locations)
            self._kdtree = KDTree(locations, leaf_size=30, metric="euclidean")
        q_loc = self._poses[k][np.newaxis, :3, 3]
        neighbor_indices = self._kdtree.query_radius(q_loc, r=distance)[0]
        return [self._keys[i] for i in neighbor_indices]

    @property
    def renderer(self):
        if not hasattr(self, "_renderer"):
            self._renderer = TorchMeshRenderer(self.device, 256)
        return self._renderer

    @property
    def pcd_renderer(self):
        if not hasattr(self, "_pcd_renderer"):
            self._pcd_renderer = Torch3DRenderer(
                self.img_size, device=self.device, radius=3.0
            )
        return self._pcd_renderer

    @property
    def fuser(self):
        if not hasattr(self, "_fuser"):
            fuser = Open3DFuser()
            keys = self._get_keys()
            fuser = Open3DFuser()
            for key in tqdm(sorted(keys)):
                rgb, depth, pose = self.get_rgbd_pose(key)
                fuser.fuse_frames(
                    torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0) / 255,
                    torch.tensor(depth).unsqueeze(0),
                    torch.tensor(pose).unsqueeze(0),
                )
                self._fuser = fuser
        return self._fuser

    @property
    def mesh(self):
        if self._mesh is None:
            mesh = self.fuser.get_mesh()
            o3d.io.write_triangle_mesh(self.mesh_path, mesh)
            self._mesh = TorchMeshRenderer.o3d_mesh_to_torch(mesh, "cuda")
        return self._mesh

    def setup(self, save_img=False):
        """
        load or compute score
        """
        score_file = os.path.join(self.output_dir, "scores.json")
        if os.path.exists(score_file):
            self._scores = json.load(open(score_file, "r"))
            return

        for key in tqdm(self._keys):
            score, depth_error = self._get_score(key)
            if save_img:
                os.makedirs(os.path.join(self.output_dir, "cache_img"), exist_ok=True)
                path = os.path.join(
                    self.output_dir,
                    "cache_img",
                    f"{key}_{score['img_psnr']:.4f}_"
                    f"{score['depth_psnr']:.4f}_{score['img_ssim']:.4f}_"
                    f"{score['depth_error']:05.2f}.jpg",
                )
                rgb = self._rgbs[key] / 255
                r_rgb = self._r_rgbs[key].numpy() / 255

                depth = self._depths[key]
                r_depth = self._r_depths[key].numpy()
                depth_cat = repeat([depth, r_depth], "n h w -> h (n w) c", c=3)
                depth_cat /= depth_cat.max()

                depth_error = repeat(
                    depth_error.astype(np.float32), "h w -> h w c", c=3
                )
                all_cat = np.concatenate([rgb, r_rgb, depth_cat, depth_error], axis=1)
                plt.imsave(path, all_cat)

        json.dump(self._scores, open(score_file, "w"))

    def _get_score(self, key, depth_error_thr=0.2, min_depth=0.1, max_depth_error=1.0):
        assert self.mesh is not None
        rgb, depth, pose = self.get_rgbd_pose(key)
        depth = torch.tensor(depth)

        r_rgb, r_depth = self.get_render_rgbd(key)

        mask = depth > min_depth
        img_psnr, _ = psnr(r_rgb[None], torch.tensor(rgb)[None], mask[None])
        img_ssim = structural_similarity(
            rgb, r_rgb.numpy().astype(np.uint8), channel_axis=2
        )

        depth_psnr, _ = psnr(r_depth[None], depth[None], max_val=5.0)
        depth_error = torch.abs(r_depth - depth)
        depth_error.clamp_(max=max_depth_error)
        depth_error_mask = depth_error > depth_error_thr
        depth_error_score = depth_error[depth_error_mask]
        depth_error_size = len(depth_error_score)
        if len(depth_error_score) > 0:
            depth_error_score = depth_error_score.mean()
        else:
            depth_error_score = 0

        score = {
            "img_psnr": float(img_psnr),
            "depth_psnr": float(depth_psnr),
            "img_ssim": float(img_ssim),
            "depth_error": float(depth_error_score),
            "depth_error_size": depth_error_size,
        }
        self._scores[key] = score
        return score, depth_error_mask.numpy()

    def get_low_quality_frames(
        self, img_psnr_thr=17.0, depth_error_thr=0.4, depth_error_size_thr=10000
    ):
        """
        filter frames with low quality
        based on the score
        """
        if self._low_quality_frames is not None:
            return self._low_quality_frames
        scores = self._scores
        foreground_frames = set(self.foreground_frames)
        # choose from color score
        color_list = set([k for k, v in scores.items() if v["img_psnr"] < img_psnr_thr])

        # choose from depth score
        depth_list = set(
            [
                k
                for k, v in scores.items()
                if v["depth_error"] > depth_error_thr
                and v["depth_error_size"] > depth_error_size_thr
            ]
        )

        candidates = (color_list | depth_list) & foreground_frames
        candidates = sorted(list(candidates))
        self._low_quality_frames = candidates
        path = os.path.join(self.output_dir, "low_quality_frames.json")
        json.dump(candidates, open(path, "w"))
        return candidates

    def get_correspondence(self, keys, depth_error_thr=0.2, correspondence_thr=0.2):
        """
        for each key in the given keys, find all the keys that have high correspondence

        NOTE: also cache the correspondence score
        """
        # key -> [corresponding keys]
        corres = defaultdict(list)

        for key in tqdm(keys):
            candidates = self.get_neighbors(key)
            for cand in candidates:
                if cand == key:
                    continue
                score = self.get_correspondence_score(key, cand, depth_error_thr)
                if score > correspondence_thr:
                    corres[key].append(cand)
        return corres

    def get_correspondence_score(self, key, cand, depth_error_thr=0.2):
        """
        warp the cand to key, and compute the size of the overlapping region in the key image

        depth_error_thr: the threshold for the depth error in meter
        """
        cand_key = "_".join((cand, key))
        if cand_key in self._correspondence:
            return self._correspondence[cand_key]
        _, _, c_pose = self.get_rgbd_pose(cand)
        _, _, k_pose = self.get_rgbd_pose(key)
        c_rgb, c_depth = self.get_render_rgbd(cand)
        k_rgb, k_depth = self.get_render_rgbd(key)

        c_rgb_warp, c_depth_warp = render_target_view_torch3d(
            input_image=c_rgb.to(device=self.device, dtype=torch.float32)[None],
            input_depth=c_depth.to(self.device)[None],
            pose_in=torch.tensor(c_pose, device=self.device)[None],
            pose_out=torch.tensor(k_pose, device=self.device)[None],
            FOV=90,
            is_3dfront=True,
            torch_renderer=self.pcd_renderer,
        )
        c_depth_warp = c_depth_warp[0].cpu()
        corres_mask = torch.abs(c_depth_warp - k_depth) < depth_error_thr
        corres_score = float(corres_mask.float().mean())
        self._correspondence[cand_key] = corres_score
        return corres_score

    def save_correspondence_score(self):
        path = os.path.join(self.output_dir, "correspondence_score.json")
        json.dump(self._correspondence, open(path, "w"))


def save_rgbd_db(
    key,
    rgb,
    depth,
    pose,
    db,
):
    """
    rgb hwc np [0,255] uint8
    depth hw in milimeter uint16
    """
    out_txn = db.begin(write=True)

    rgb_key = f"{key}_rgb".encode("ascii")
    rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    out_txn.put(rgb_key, rgb)

    depth_key = f"{key}_depth".encode("ascii")
    depth = zlib.compress(depth.tobytes())
    out_txn.put(depth_key, depth)

    pose_key = f"{key}_pose".encode("ascii")
    pose = zlib.compress(pose.astype(np.float32).tobytes())
    out_txn.put(pose_key, pose)

    out_txn.commit()


def save_rgbd_img(
    frames_meta,
    imgs,
    depths,
    out_dir,
):
    for img, depth, frame_id in zip(imgs, depths, frames_meta):
        img_path = f"{out_dir}/{frame_id}.jpeg"
        depth_path = f"{out_dir}/{frame_id}.png"
        plt.imsave(img_path, img)
        cv2.imwrite(depth_path, depth)


def generate_ddim_inversion(
    pipeline,
    batch,
    weight_dtype,
    generator,
    accelerator,
    depth_transform,
    warp_img,
    warp_depth,
    guidance_scale,
    use_ray=True,
    output_depth=False,
    ignore_prompts=False,
):
    """
    warp_img: torch float32 [0,255] thwc
    warp_depth: torch float32 in meter thw
    """
    T_in = batch["image_input"].size(1)
    T_out = batch["image_target"].size(1)
    # gt_image = batch["image_target"].to(dtype=weight_dtype).to(accelerator.device)
    input_image = batch["image_input"].to(dtype=weight_dtype).to(accelerator.device)
    pose_in = batch["pose_in"].to(dtype=weight_dtype).to(accelerator.device)  # BxTx4
    pose_out = batch["pose_out"].to(dtype=weight_dtype).to(accelerator.device)  # BxTx4
    pose_in_inv = (
        batch["pose_in_inv"].to(dtype=weight_dtype).to(accelerator.device)
    )  # BxTx4
    pose_out_inv = (
        batch["pose_out_inv"].to(dtype=weight_dtype).to(accelerator.device)
    )  # BxTx4

    input_depth = batch["depth_input"].to(accelerator.device)
    # NOTE assume batch size is 1
    # warp_img, warp_depth = render_target_view_torch3d(
    #     rearrange((input_image + 1) * (255 / 2), "b t c h w -> (b t) h w c").to(
    #         torch.float32
    #     ),
    #     rearrange(input_depth, "b t h w -> (b t) h w").to(torch.float32),
    #     rearrange(pose_in.to(torch.float32), "b t c d -> (b t) c d"),
    #     rearrange(pose_out.to(torch.float32), "b t c d -> (b t) c d"),
    #     FOV=90.0,
    #     is_3dfront=True,
    #     torch_renderer=torch_renderer,
    #     batch_size=1,
    # )
    warp_img = rearrange(warp_img, "t h w c -> 1 t c h w")
    warp_img = (warp_img.to(dtype=weight_dtype) / 255.0 - 0.5) * 2
    warp_depth = [depth_transform(depth) for depth in warp_depth]
    warp_depth = rearrange(warp_depth, "t h w -> 1 t h w").to(dtype=weight_dtype)
    torch.cuda.empty_cache()
    # gt_image = rearrange(gt_image, "b t c h w -> (b t) c h w")
    input_image = rearrange(input_image, "b t c h w -> (b t) c h w")  # T_in

    kwargs = {}
    if use_ray:
        target_rays = [
            v.to(dtype=weight_dtype, device=accelerator.device)
            for k, v in batch.items()
            if "target_ray" in k
        ]
        kwargs["target_rays"] = target_rays

        cond_rays = [
            v.to(dtype=weight_dtype, device=accelerator.device)
            for k, v in batch.items()
            if "cond_ray" in k
        ]
        cond_rays = [rearrange(r, "b t c h w -> (b t) c h w") for r in cond_rays]

        if len(cond_rays):
            # NOTE: only support 1 cond_ray
            kwargs["cond_rays"] = cond_rays[0]
    # prepare layout if any
    if "layout_cls" in batch:
        layout_pos = batch["layout_pos"].to(
            dtype=weight_dtype, device=accelerator.device
        )
        kwargs["layouts"] = {
            "layout_pos": layout_pos,
            "layout_cls": batch["layout_cls"].to(device=accelerator.device),
        }
    if "back_layout_cls" in batch:
        kwargs["layouts"]["back_layout_cls"] = batch["back_layout_cls"].to(
            device=accelerator.device
        )
        kwargs["layouts"]["back_layout_pos"] = batch["back_layout_pos"].to(
            dtype=weight_dtype, device=accelerator.device
        )

    in_pos3d = batch.get("in_pos3d", None)
    if in_pos3d is not None:
        in_pos3d = in_pos3d.to(dtype=weight_dtype, device=accelerator.device)

    h, w = input_image.shape[2:]
    with torch.autocast("cuda"):
        image = pipeline(
            input_imgs=input_image,
            prompt_imgs=input_image,
            poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
            height=h,
            width=w,
            T_in=T_in,
            T_out=pose_out.shape[1],
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            generator=generator,
            output_type="numpy",
            output_depth=output_depth,
            in_pos3d=in_pos3d,
            warp_img=warp_img,
            warp_depth=warp_depth,
            ddim_inversion=True,
            ignore_prompts=ignore_prompts,
            **kwargs,
        ).images

        # t c h w
        pred_image = (
            torch.from_numpy(image * 2.0 - 1.0)
            .permute(0, 3, 1, 2)
            .to(accelerator.device)
        )
    return pred_image


def regenerate(
    cfg,
    view_sets,
    mesh_render_db_path,
    output_dir,
    refinement,
    scene_id,
    accelerator,
    pipeline,
    depth_model,
    max_view=60,
    weight_dtype=torch.float16,
    max_depth=29.9,
    min_depth=0.1,
):
    """ """
    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
    # go back one level in db path
    dataset = make_dataset(cfg, os.path.dirname(mesh_render_db_path), scene_id)

    depth_func = lambda x: depth_model.infer(x[None, ...])["depth"][0, 0]
    depth_transform = instantiate_from_config(cfg.data.params.depth_tform_cfg)

    sequence = []
    for view, correspondences in view_sets.items():
        frames_meta = {
            "cond": [sorted(correspondences)[0]],
            "target": correspondences[:max_view] + [view],
        }
        batch = dataset.get_item_from_meta(scene_id, frames_meta)
        # except:
        #     print(scene_id, frames_meta)
        #     print(dataset.root_dir)
        #     exit()
        batch.update(
            dataset.get_layout(
                scene_id,
                frames_meta["target"],
                batch["pose_out"],
                dataset.layout_dir,
            )
        )
        batch = collate_fn([batch])
        warp_img, warp_depth = [], []
        for view_id in frames_meta["target"]:
            rgb, depth = refinement.get_render_rgbd(view_id)
            warp_img.append(rgb)
            warp_depth.append(depth)
        warp_img = torch.stack(warp_img).to(accelerator.device)
        warp_depth = torch.stack(warp_depth).to(accelerator.device)
        pred_imgs = generate_ddim_inversion(
            pipeline,
            batch,
            weight_dtype,
            generator,
            accelerator,
            depth_transform,
            warp_img=warp_img,
            warp_depth=warp_depth,
            guidance_scale=cfg.model.guidance_scale,
            use_ray=True,
            output_depth=True,
            # ignore_prompts=True,
        )
        imgs, r_depths = process_output(pred_imgs)
        a_depths, methods = get_absolute_depth(
            imgs=rearrange(imgs, "t h w c -> t c h w"),
            relative_depths=r_depths,
            frames_meta=frames_meta["target"],
            depth_model=depth_func,
            layout_db_path=os.path.join(dataset.layout_dir, scene_id),
        )
        sequence.append({"frame_ids": frames_meta, "depth_method": methods})
        invalid_a_depth_mask = (a_depths > max_depth) | (a_depths < min_depth)
        a_depths[invalid_a_depth_mask] = 0.0
        a_depths = a_depths * 1000
        assert a_depths.max() < 2**16
        a_depths = a_depths.cpu().numpy()
        a_depths = np.round(a_depths).astype(np.uint16)

        imgs = imgs.cpu().numpy()
        imgs = np.round(imgs).astype(np.uint8)
        img_dir = os.path.join(output_dir, view)
        os.makedirs(img_dir, exist_ok=True)
        save_rgbd_img(frames_meta["target"], imgs, a_depths, img_dir)
    json.dump(sequence, open(f"{output_dir}/sequence.json", "w"))


def process_output(output):
    output = rearrange(output, "(t n) c h w -> n t h w c", n=2)
    img = output[0]
    img = (img + 1) / 2.0 * 255.0
    img = torch.clamp(img, min=0.0, max=255.0)

    depth = output[1]
    depth = depth.mean(-1)
    depth = (depth + 1) * 0.5
    return img, depth  # t h w c and t h w


def re_fuse(db_path, db_keys, regen_path, out_file):
    db = DB(db_path, True)
    regen_path = list(glob(f"{regen_path}/*/*.jpeg"))
    regen_path = {os.path.basename(x).split(".")[0]: x for x in regen_path}
    print(f"got {len(regen_path)} regen images")
    fuser = Open3DFuser(fusion_resolution=0.01)
    for key, path in tqdm(regen_path.items()):
        img = plt.imread(path)
        img = rearrange(img, "h w c -> 1 c h w")
        img = torch.tensor(img).to(torch.float32) / 255
        depth = cv2.imread(path.replace(".jpeg", ".png"), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000
        depth = torch.tensor(depth).unsqueeze(0)
        pose = db.get_pose(key)
        pose = torch.tensor(pose).unsqueeze(0)
        fuser.fuse_frames(img, depth, pose)

    print(f"got {len(set(db_keys))} db images")
    db_key = set(db_keys) - set(regen_path.keys())
    print(f"refuse {len(db_key)} images after removing regenerated images")
    # for key in tqdm(db_key):
    #     img = db.get_rgb(key)
    #     img = rearrange(img, "h w c -> 1 c h w")
    #     img = torch.tensor(img).to(torch.float32) / 255

    #     depth = db.get_depth(key)
    #     depth = torch.tensor(depth).unsqueeze(0)

    #     pose = db.get_pose(key)
    #     pose = torch.tensor(pose).unsqueeze(0)
    #     fuser.fuse_frames(img, depth, pose)

    fuser.export_mesh(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    data_root = "/work/vig/hieu/3dfront_data"
    cfgs = [
        "./configs/base.yaml",
        "./configs/base_layout_rcn_iodepth.yaml",
        "./configs/layout_3dfront_random_pose_iodepth_affine_1871_scene_0.001quantile.yaml",
    ]
    ckpt_path = "/work/vig/hieu/escher/logs_eschernet_layout_3dfront_random_pose_iodepth_affine1871_1e-5quantile/pipeline-21000"

    configs = [OmegaConf.load(cfg) for cfg in cfgs]
    cfg = OmegaConf.merge(*configs)

    cfg.data.params.val_scene_ids = f"{data_root}/val_scenes_300_3000.json"
    cfg.data.params.layout_dir = f"{data_root}/layout_pcd_3000scenes_random_floor"
    cfg.data.params.val_dir = f"{data_root}/images_3000scenes_random_floor"

    cfg.data.params.return_depth_input = True  # ddim inversion
    weight_dtype = torch.float16
    image_size = 256

    val_scene_ids = sorted(json.load(open(cfg.data.params.val_scene_ids, "r")))
    val_scene_ids = val_scene_ids[args.start : args.end]
    val_scene_ids = val_scene_ids[args.offset :: args.step]
    exp_name = "iodepth_affine1871_21k"
    gen_root = f"/work/vig/hieu/gen300/{exp_name}"
    refinement_root = f"/work/vig/hieu/refinement300/{exp_name}"
    for scene_id in tqdm(val_scene_ids):
        if not os.path.exists(f"{gen_root}/sequence_{scene_id}.json"):
            print(f"skip {scene_id} generation incomplete")
            continue
        if os.path.exists(f"{refinement_root}/{scene_id}/first_round_mesh.ply"):
            print(f"skip {scene_id} refinement already done")
            continue
        json_path = f"{gen_root}/sequence_{scene_id}.json"
        db_path = f"{gen_root}/db/{scene_id}"
        layout_db_path = f"{data_root}/layout_pcd_3000scenes_random_floor/{scene_id}"
        out_dir = f"{refinement_root}/{scene_id}"
        mesh_path = f"{out_dir}/first_round_mesh.ply"
        mesh_render_db_path = f"{out_dir}/first_round_render_db/{scene_id}"

        os.makedirs(out_dir, exist_ok=True)
        refinement = Refinement(mesh_path, db_path, layout_db_path, out_dir, json_path)
        refinement.mesh
        # keys = refinement._get_keys()
        # refinement.setup(True)
        # for key in keys:
        #     refinement.get_rgbd_pose(key)

        # candidates = refinement.get_low_quality_frames()
        # corres = refinement.get_correspondence(candidates, depth_error_thr=1.0)
        # refinement.save_correspondence_score()
        # corres_path = os.path.join(out_dir, "correspondence_frames.json")
        # json.dump(corres, open(corres_path, "w"))
        # os.makedirs(mesh_render_db_path, exist_ok=True)
        # refinement.make_render_db(mesh_render_db_path)

        # accelerator = Accelerator(
        #     mixed_precision=cfg.training.mixed_precision,
        # )
        # pipeline = make_pipeline(
        #     cfg, ckpt_path, weight_dtype, accelerator, inverse_ddim=True
        # )
        # depth_model = torch.hub.load(
        #     "lpiccinelli-eth/UniDepth",
        #     "UniDepth",
        #     version="v1",
        #     backbone="ViTL14",
        #     pretrained=True,
        #     trust_repo=True,
        #     # force_reload=True,
        # ).to(accelerator.device)
        # regen_out = f"{out_dir}/round2"
        # regenerate(
        #     cfg,
        #     corres,
        #     mesh_render_db_path,
        #     regen_out,
        #     refinement,
        #     scene_id,
        #     accelerator,
        #     pipeline,
        #     depth_model,
        # )

        # re_fuse(db_path, keys, regen_out, f"{out_dir}/round2_regen.ply")


"""
CUDA_VISIBLE_DEVICES=1 python pipeline.py  --step 2 --offset 1
"""
