import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from typing import List, Dict, Tuple
import zlib
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange
from torchvision import transforms
from cfg_util import get_obj_from_str

from .data_utils import (
    get_world_pcd,
    collate_fn,
)
from .mp import RCNMatterport, RCNMatterportExt, get_c2w
import lmdb


class LayoutRCNMatterportDataModule:
    def __init__(
        self,
        image_strides: List[int],
        batch_size,
        graph_dir,
        train_dir,
        dataset_cls: str,
        val_dir=None,
        image_size=256,
        available_pano_file: str = None,
        max_rot_step=4,
        min_rot_step=2,
        dist_threshold=3.0,  # in meters
        num_workers=4,
        cond_image_strides=[],
        cond_image_size=224,
        # layout args
        layout_shape_db=(512, 512),
        layout_dir: str = "",
        max_num_points=20,
        intersection_pos="height",
        use_back_point=False,
        back_point_dir=None,
        layout_depth_scale=2000.0,
    ):

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.graph_dir = graph_dir
        self.num_workers = num_workers
        self.image_strides = image_strides
        self.image_size = image_size
        self.max_rot_step = max_rot_step
        self.min_rot_step = min_rot_step
        self.available_pano_file = available_pano_file
        self.dist_threshold = dist_threshold
        self.dataset_cls = dataset_cls
        self.cond_image_strides = cond_image_strides
        self.cond_image_size = cond_image_size

        # layout args
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos
        self.use_back_point = use_back_point
        self.back_point_dir = back_point_dir
        self.layout_depth_scale = layout_depth_scale

        assert dataset_cls.split(".")[-1] in [
            "LayoutRCNMatterport",
            "LayoutRCNMatterportExt",
        ]
        self.tform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def train_dataloader(self):
        dataset = get_obj_from_str(self.dataset_cls)(
            image_strides=self.image_strides,
            available_pano_file=self.available_pano_file,
            graph_dir=self.graph_dir,
            root_dir=self.train_dir,
            image_transforms=self.tform,
            image_size=self.image_size,
            mode="train",
            max_rot_step=self.max_rot_step,
            min_rot_step=self.min_rot_step,
            dist_threshold=self.dist_threshold,
            cond_image_strides=self.cond_image_strides,
            cond_image_size=self.cond_image_size,
            # layout args
            layout_shape_db=self.layout_shape_db,
            layout_dir=self.layout_dir,
            max_num_points=self.max_num_points,
            intersection_pos=self.intersection_pos,
            use_back_point=self.use_back_point,
            back_point_dir=self.back_point_dir,
            layout_depth_scale=self.layout_depth_scale,
        )
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def train_log_dataloader(self):
        dataset = get_obj_from_str(self.dataset_cls)(
            image_strides=self.image_strides,
            available_pano_file=self.available_pano_file,
            graph_dir=self.graph_dir,
            root_dir=self.train_dir,
            image_transforms=self.tform,
            image_size=self.image_size,
            mode="val",
            max_rot_step=self.max_rot_step,
            min_rot_step=self.min_rot_step,
            dist_threshold=self.dist_threshold,
            cond_image_strides=self.cond_image_strides,
            cond_image_size=self.cond_image_size,
            # layout args
            layout_shape_db=self.layout_shape_db,
            layout_dir=self.layout_dir,
            max_num_points=self.max_num_points,
            intersection_pos=self.intersection_pos,
            use_back_point=self.use_back_point,
            back_point_dir=self.back_point_dir,
            layout_depth_scale=self.layout_depth_scale,
        )
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        assert self.val_dir is not None
        dataset = get_obj_from_str(self.dataset_cls)(
            image_strides=self.image_strides,
            available_pano_file=self.available_pano_file,
            graph_dir=self.graph_dir,
            root_dir=self.val_dir,
            image_transforms=self.tform,
            image_size=self.image_size,
            mode="val",
            max_rot_step=self.max_rot_step,
            min_rot_step=self.min_rot_step,
            dist_threshold=self.dist_threshold,
            cond_image_strides=self.cond_image_strides,
            cond_image_size=self.cond_image_size,
            # layout args
            layout_shape_db=self.layout_shape_db,
            layout_dir=self.layout_dir,
            max_num_points=self.max_num_points,
            intersection_pos=self.intersection_pos,
            use_back_point=self.use_back_point,
            back_point_dir=self.back_point_dir,
            layout_depth_scale=self.layout_depth_scale,
        )
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            collate_fn=collate_fn,
        )


class LayoutRCNMatterport(RCNMatterport):
    def __init__(
        self,
        image_strides: List[int],
        graph_dir,
        root_dir,
        image_transforms=[],
        image_size=256,
        mode="train",
        max_rot_step=4,
        min_rot_step=2,
        available_pano_file=None,
        dist_threshold=3.0,  # in meters
        cond_image_strides=[],
        cond_image_size=224,
        target_dir=None,
        # layout args
        layout_shape_db=(512, 512),
        layout_dir: str = "",
        max_num_points=20,
        intersection_pos="height",
        use_back_point=False,
        back_point_dir=None,
        layout_depth_scale=2000.0,
    ):
        super().__init__(
            image_strides=image_strides,
            graph_dir=graph_dir,
            root_dir=root_dir,
            image_transforms=image_transforms,
            image_size=image_size,
            mode=mode,
            max_rot_step=max_rot_step,
            min_rot_step=min_rot_step,
            available_pano_file=available_pano_file,
            dist_threshold=dist_threshold,
            cond_image_strides=cond_image_strides,
            cond_image_size=cond_image_size,
            target_dir=target_dir,
        )
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos
        self.use_back_point = use_back_point
        self.back_point_dir = back_point_dir
        self.layout_depth_scale = layout_depth_scale

        self.LAYOUT_K = self.K / min(image_strides)
        self.LAYOUT_K[2, 2] = 1.0

    def __getitem__(self, index):
        scene_id, pano_ids = self.items_meta[index]
        frames_meta = self._sample_frames_meta(pano_ids)
        data = self.get_item_from_meta(scene_id, frames_meta)

        # get the layout
        data.update(
            self.get_layout(
                scene_id,
                frames_meta["target"],
                self.layout_dir,
            )
        )
        if self.use_back_point:
            back_point_layout = self.get_layout(
                scene_id,
                frames_meta["target"],
                self.back_point_dir,
            )
            back_point_layout = {f"back_{k}": v for k, v in back_point_layout.items()}
            data.update(back_point_layout)

        return data

    def get_layout(
        self,
        scene_id: str,
        frames_meta: List[Tuple[str, int]],
        layout_dir: str,
    ) -> Dict:

        H, W = self.layout_shape_db
        depth_scale = self.layout_depth_scale
        assert H == W and H % self.image_size == 0
        shape_scale = H // self.image_size * min(self.image_strides)

        layout = {"layout_cls": [], "layout_pos": []}

        env = lmdb.open(
            os.path.join(layout_dir, scene_id),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        txn = env.begin(write=False)
        for pano_id, frame_id in frames_meta:
            # load layout condition
            key = f"{scene_id}_{pano_id}_{frame_id:0>2}".encode("ascii")
            data = txn.get(key)
            data = zlib.decompress(data)
            data = np.frombuffer(data, dtype=np.uint16).reshape(-1, 2, H, W)
            # NOTE flip to fix bug in layout preprocessing
            data = np.ascontiguousarray(np.flip(data, axis=(-1, -2)))
            layout_cls = torch.tensor(
                data[
                    :,
                    0,
                    ...,
                    shape_scale // 2 :: shape_scale,
                    shape_scale // 2 :: shape_scale,
                ].astype(np.int32),
                dtype=torch.long,
            )
            depth = torch.tensor(
                data[
                    :,
                    1,
                    ...,
                    shape_scale // 2 :: shape_scale,
                    shape_scale // 2 :: shape_scale,
                ].astype(np.float32)
                / depth_scale
            )

            # sort by depth to cutoff the far away points when using  max_num_points
            mask = depth < 0.1
            depth[mask] = 10000.0
            indices = torch.argsort(depth, dim=0)  # n,h,w
            layout_cls = torch.take_along_dim(layout_cls, indices, dim=0)
            depth = torch.take_along_dim(depth, indices, dim=0)
            mask = torch.take_along_dim(mask, indices, dim=0)
            depth[mask] = 0.0
            depth = depth[: self.max_num_points]
            layout_cls = layout_cls[: self.max_num_points]

            layout["layout_cls"].append(
                layout_cls
                # layout_cls[
                #     ...,
                #     shape_scale // 2 :: shape_scale,
                #     shape_scale // 2 :: shape_scale,
                # ]
            )

            # compute position of intersection point
            # only consider the relative height of the camera wrt the floor
            # and elevation of the camera
            # In this dataset, the camera always looks horizontally,
            # and the camera heights are roughly the same,
            # so we can directly use the camera coordinates to compute the intersection point
            P = torch.eye(4, dtype=torch.float32)
            # depth = depth[
            #     ..., shape_scale // 2 :: shape_scale, shape_scale // 2 :: shape_scale
            # ]
            pos_3d = get_world_pcd(P, self.LAYOUT_K, depth)
            if self.intersection_pos == "height":
                layout["layout_pos"].append(
                    torch.stack([depth, pos_3d[..., -1]], dim=1)
                )
            elif self.intersection_pos == "3d":
                depth = rearrange(depth, "n h w -> n 1 h w")
                pos_3d = rearrange(pos_3d, "n h w c -> n c h w")
                layout["layout_pos"].append(torch.cat([depth, pos_3d], dim=1))
        env.close()

        layout["layout_pos"] = pad_sequence(layout["layout_pos"], batch_first=True)
        layout["layout_cls"] = pad_sequence(layout["layout_cls"], batch_first=True)
        return layout


class LayoutRCNMatterportExt(RCNMatterportExt, LayoutRCNMatterport):
    def __init__(
        self,
        image_strides: List[int],
        graph_dir,
        root_dir,
        image_transforms=[],
        image_size=256,
        mode="train",
        max_rot_step=2,
        min_rot_step=1,
        available_pano_file=None,
        dist_threshold=3.0,  # in meters
        cond_image_strides=[],
        cond_image_size=224,
        # layout args
        layout_shape_db=(256, 256),
        layout_dir: str = "",
        max_num_points=20,
        intersection_pos="height",
        use_back_point=False,
        back_point_dir=None,
        layout_depth_scale=2000.0,
    ):
        super().__init__(
            image_strides=image_strides,
            graph_dir=graph_dir,
            root_dir=root_dir,
            image_transforms=image_transforms,
            image_size=image_size,
            mode=mode,
            max_rot_step=max_rot_step,
            min_rot_step=min_rot_step,
            available_pano_file=available_pano_file,
            dist_threshold=dist_threshold,
            cond_image_strides=cond_image_strides,
            cond_image_size=cond_image_size,
        )
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos
        self.use_back_point = use_back_point
        self.back_point_dir = back_point_dir
        self.layout_depth_scale = layout_depth_scale

        self.LAYOUT_K = self.K / min(image_strides)
        self.LAYOUT_K[2, 2] = 1.0

    def __getitem__(self, index):
        return LayoutRCNMatterport.__getitem__(self, index)

    def get_layout(
        self,
        scene_id: str,
        frames_meta: List[Tuple[str, int]],
        layout_dir: str,
    ) -> Dict:

        H, W = self.layout_shape_db
        depth_scale = self.layout_depth_scale
        assert H == W and H % self.image_size == 0
        shape_scale = H // self.image_size * min(self.image_strides)

        layout = {"layout_cls": [], "layout_pos": []}

        env = lmdb.open(
            os.path.join(layout_dir, scene_id),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        txn = env.begin(write=False)
        for pano_id, frame_id in frames_meta:
            # load layout condition
            key = f"{scene_id}_{pano_id}_{frame_id:0>2}".encode("ascii")
            data = txn.get(key)
            data = zlib.decompress(data)
            data = np.frombuffer(data, dtype=np.uint16).reshape(-1, 2, H, W)
            # flip to fix bug in layout preprocessing
            data = np.ascontiguousarray(np.flip(data, axis=(-1, -2)))

            layout_cls = torch.tensor(
                data[
                    :,
                    0,
                    ...,
                    shape_scale // 2 :: shape_scale,
                    shape_scale // 2 :: shape_scale,
                ].astype(np.int32),
                dtype=torch.long,
            )
            depth = torch.tensor(
                data[
                    :,
                    1,
                    ...,
                    shape_scale // 2 :: shape_scale,
                    shape_scale // 2 :: shape_scale,
                ].astype(np.float32)
                / depth_scale
            )

            # sort by depth to cutoff the far away points when using  max_num_points
            mask = depth < 0.1
            depth[mask] = 10000.0
            indices = torch.argsort(depth, dim=0)  # n,h,w
            layout_cls = torch.take_along_dim(layout_cls, indices, dim=0)
            depth = torch.take_along_dim(depth, indices, dim=0)
            mask = torch.take_along_dim(mask, indices, dim=0)
            depth[mask] = 0.0
            depth = depth[: self.max_num_points]
            layout_cls = layout_cls[: self.max_num_points]

            layout["layout_cls"].append(
                layout_cls
                # layout_cls[
                #     ...,
                #     shape_scale // 2 :: shape_scale,
                #     shape_scale // 2 :: shape_scale,
                # ]
            )

            # compute position of intersection point
            # only consider the relative height of the camera wrt the floor
            # and elevation of the camera
            # In this dataset, the camera heights are roughly the same,
            # so we only include camera elevation in camera pose
            ## get camera elevation
            elevation = self.ELEVATIONS[frame_id // len(self.HEADINGS)]
            pose = np.array([0.0, 0.0, 0.0, 0.0, elevation])
            P = get_c2w(pose)
            # depth = depth[
            #     ..., shape_scale // 2 :: shape_scale, shape_scale // 2 :: shape_scale
            # ]
            pos_3d = get_world_pcd(P, self.LAYOUT_K, depth)
            if self.intersection_pos == "height":
                layout["layout_pos"].append(
                    torch.stack([depth, pos_3d[..., -1]], dim=1)
                )
            elif self.intersection_pos == "3d":
                depth = rearrange(depth, "n h w -> n 1 h w")
                pos_3d = rearrange(pos_3d, "n h w c -> n c h w")
                layout["layout_pos"].append(torch.cat([depth, pos_3d], dim=1))
        env.close()

        layout["layout_pos"] = pad_sequence(layout["layout_pos"], batch_first=True)
        layout["layout_cls"] = pad_sequence(layout["layout_cls"], batch_first=True)
        return layout
