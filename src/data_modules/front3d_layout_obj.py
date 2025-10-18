"""
using 1 vector per object for layout representation

"""

import json
import os
import pickle
import zlib
from itertools import chain, zip_longest
from typing import Dict, List, Set, Tuple

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .data_utils import get_world_pcd
from .front3d import FRONT3D_BASE_MATRIX, Front3DLocationGraph, Front3DPose
from .front3d_combine import Front3DObjPose


class ObjLayoutMixin:
    """
    assumption:
        the right direction of the camera is horizontal
        the up direction of the world coordinate is Y-axis
        the camera coordinate is Y up, X right, Z backward
    """

    def get_obj_layout(
        self, poses, obj_layouts: List[Dict], box_range: float, n_max: int
    ) -> Dict:
        """
        load furniture object from the layout
        each obj has class, center, size, and orientation
        transform object pose to the 2d camera coordinate (converted from 3d camera pose)
        sort from near to far

        args:
            poses: w2c (n_target,4,4)
            obj_layouts: list of dict
                {
                    bbox: (4,2) 4 vertices x,z coordinate
                    label_id: int
                    label: str
                }
            box_range: size of the square in front of the camera to include the objects (in meter)

        return dict {
                layout_obj_cls: (n_target, n)
                layout_obj_pos: (n_target, n, 4, 2)
                layout_obj_camheight: (n_target,)
                layout_obj_camelevation: (n_target,)
            }
        """
        c2w = torch.linalg.inv(poses)
        c2w_2d, height, elevation = self.get_2d_pose(c2w)
        # if no object, return dummy layout
        if len(obj_layouts) == 0:
            return {
                "layout_obj_cls": torch.zeros((len(poses), 1), dtype=torch.long),
                "layout_obj_pos": torch.zeros(
                    (len(poses), 1, 4, 2), dtype=torch.float32
                ),
                "layout_obj_camheight": height,
                "layout_obj_camelevation": elevation,
            }

        out = {"layout_obj_cls": [], "layout_obj_pos": []}
        w2c_2d = torch.linalg.inv(c2w_2d)
        obj_classes = torch.tensor(
            [obj["label_id"] for obj in obj_layouts], dtype=torch.long
        )
        obj_boxes = [obj["bbox"] for obj in obj_layouts]
        # n,4,2 (x,z) then flip to (z,x)
        obj_boxes = np.stack(obj_boxes, axis=0)[..., [1, 0]]  
        obj_boxes = torch.tensor(obj_boxes, dtype=torch.float32)
        for pose in w2c_2d:
            cam_boxes = obj_boxes @ pose[:2, :2].T + pose[:2, 2]

            # filter boxes within the range
            center = cam_boxes.mean(dim=1)
            inliers = (
                (center[..., 1] > 0)
                & (center[..., 1] < box_range)
                & (center[..., 0] > -box_range * 0.5)
                & (center[..., 0] < box_range * 0.5)
            )
            center = center[inliers]
            cam_boxes = cam_boxes[inliers]
            cam_classes = obj_classes[inliers]

            if len(center) > 0:
                # sort by distance
                indices = torch.argsort(center.norm(dim=1))
                cam_boxes = cam_boxes[indices]
                cam_classes = cam_classes[indices]

                # limit the number of objects
                cam_boxes = cam_boxes[:n_max]
                cam_classes = cam_classes[:n_max]
            out["layout_obj_cls"].append(cam_classes)
            out["layout_obj_pos"].append(cam_boxes)
        max_n = max(len(objs) for objs in out["layout_obj_cls"])
        if max_n == 0:
            return {
                "layout_obj_cls": torch.zeros((len(poses), 1), dtype=torch.long),
                "layout_obj_pos": torch.zeros(
                    (len(poses), 1, 4, 2), dtype=torch.float32
                ),
                "layout_obj_camheight": height,
                "layout_obj_camelevation": elevation,
            }
        out["layout_obj_cls"] = pad_sequence(out["layout_obj_cls"], batch_first=True)
        out["layout_obj_pos"] = pad_sequence(out["layout_obj_pos"], batch_first=True)
        print(f"layout_obj_cls: {out['layout_obj_cls'].shape}")
        print(f"layout_obj_pos: {out['layout_obj_pos'].shape}")
        out.update(
            {
                "layout_obj_camheight": height,
                "layout_obj_camelevation": elevation,
            }
        )
        return out

    @classmethod
    def get_2d_pose(cls, pose3d: Tensor) -> Tuple[Tensor]:
        """
        convert 3d pose of camera to 2d pose in the horizontal plane (z,x plane)
        assumption:
            the right direction of the camera is horizontal
            the up direction of the world coordinate is Y-axis
            the y-component of the camera up direction is positive
            the camera coordinate is Y up, X right, Z backward

        2d pose convention (c2w_2d):
            (u,v) local coordinate
            c2w_2d@(u,v) = (z,x) world coordinate
            first component (u) is the right direction
            second component (v) is the forward direction
        args:
            pose3d: 3d pose of the camera (c2w) (n,4,4)

        return:
            pose2d: 2d pose of the camera (c2w) (n,3,3)
            height_cam3d: in meters
            elevation_cam3d: in radians, positive is up
        """
        # check the right direction is horizontal
        assert (
            pose3d[:, 1, 0].abs().max() < 1e-5
        ), f"the right direction is not horizontal, {pose3d[:, 1, 0].abs().max()}"
        pose2d = torch.zeros(
            (pose3d.size(0), 3, 3), device=pose3d.device, dtype=pose3d.dtype
        )
        pose2d[:, -1, -1] = 1.0
        right_2d = pose3d[:, [2, 0], 0]
        right_2d = right_2d / right_2d.norm(dim=1, keepdim=True)
        forward_2d = -pose3d[:, [2, 0], 2]
        forward_2d = forward_2d / forward_2d.norm(dim=1, keepdim=True)
        pose2d[:, :2, 0] = right_2d
        pose2d[:, :2, 1] = forward_2d
        pose2d[:, :2, 2] = pose3d[:, [2, 0], 3]

        height_cam3d = pose3d[:, 1, 3]

        # lengh of the projection of the forward vector to the horizontal plane
        forward_proj = pose3d[:, [0, 2], 2].norm(dim=1)
        elevation_cam3d = torch.atan2(-pose3d[:, 1, 2], forward_proj)
        return pose2d, height_cam3d, elevation_cam3d


def get_wall_layout(
    self,
    poses,
    scene_id: str = None,
    frame_ids: List[str] = None,
    layout_dir: str = None,
    layout_depths: List[torch.FloatTensor] = None,
    layout_clss: List[torch.LongTensor] = None,
) -> Dict:
    """
    load wall, floor, ceiling layout from frame meta
    modified from Front3DPose.get_layout this one
    args:
        poses: w2c
    """
    assert isinstance(self, (Front3DPose, Front3DLocationGraph, Front3DObjPose))
    H, W = self.layout_shape_db
    depth_scale = self.depth_scale
    assert H == W  # and H % self.image_size == 0
    # shape_scale = H // self.image_size * min(self.image_strides)
    layout_size = self.image_size // min(self.image_strides)

    layout = {"layout_cls": [], "layout_pos": []}

    if layout_dir is not None:
        assert scene_id is not None
        assert frame_ids is not None
        if not os.path.exists(os.path.join(layout_dir, scene_id)):
            layout_dir = layout_dir.replace("/work/vig/hieu/3dfront_data/", self.data_root + "/")
        env = lmdb.open(
            os.path.join(layout_dir, scene_id),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        txn = env.begin(write=False)
    else:
        assert layout_depths is not None
        assert layout_clss is not None
        frame_ids = [None] * len(layout_depths)

    for i, (frame_id, w2c) in enumerate(zip(frame_ids, poses)):
        # load layout condition
        if layout_dir is not None:
            key = frame_id.encode("ascii")
            data = txn.get(key)
            data = zlib.decompress(data)
            #### take the wall only
            data = np.frombuffer(data, dtype=np.uint16).reshape(-1, 2, H, W)[:1]
            data = torch.tensor(data.astype(np.int32), dtype=torch.float32)
            data = F.interpolate(data, size=layout_size, mode="nearest")
            layout_cls = data[:, 0].long()
            depth = data[:, 1] / depth_scale
        else:
            #### take the wall only
            depth = layout_depths[i][:1]
            layout_cls = layout_clss[i][:1]
            if depth.size(1) != layout_size:
                depth = F.interpolate(
                    depth[:, None], size=layout_size, mode="nearest"
                ).squeeze(1)
                layout_cls = (
                    F.interpolate(
                        layout_cls[:, None].float(),
                        size=layout_size,
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                )
        # TODO remapping cls id
        if self.enable_remapping_cls:
            layout_cls[0] = self.remapping_wall_cls(layout_cls[0])

        layout["layout_cls"].append(layout_cls)
        # get position of intersection points
        # only consider the position and orientation w.r.t the floor
        # meaning the camera elevation and camera height
        # ignoring the camera heading
        # for the camera height, here we use the height (y) in the world coordinate
        c2w = torch.linalg.inv(w2c)
        R = c2w[:3, :3]
        # TODO decompose R to remove the camera heading rotation

        # change cam coordinate
        R = R @ torch.tensor(FRONT3D_BASE_MATRIX, device=R.device)
        c2w[:3, :3] = R
        pos_3d = get_world_pcd(c2w, self.LAYOUT_K, depth)
        if self.intersection_pos == "height":
            # height is y coord
            layout["layout_pos"].append(torch.stack([depth, pos_3d[..., 1]], dim=1))
        elif self.intersection_pos == "3d":
            # raise NotImplementedError("need to remove heading from pose first")
            depth = rearrange(depth, "n h w -> n 1 h w")
            pos_3d = rearrange(pos_3d, "n h w c -> n c h w")
            layout["layout_pos"].append(torch.cat([depth, pos_3d], dim=1))
    if layout_dir is not None:
        env.close()

    layout["layout_pos"] = pad_sequence(layout["layout_pos"], batch_first=True)
    layout["layout_cls"] = pad_sequence(layout["layout_cls"], batch_first=True)
    return layout


class Front3DPoseObjLayout(Front3DPose, ObjLayoutMixin):
    """
    Front3DPose dataset with object layout
    """

    def __init__(
        self,
        image_strides: List[int],
        graph_dir,
        root_dir,
        scene_ids: List[str],
        pose_dir=None,
        image_transforms=[],
        image_size=256,
        mode="train",
        cond_image_strides=[],
        cond_image_size=224,
        target_dir=None,
        n_cond_frames=3,
        n_target_frames=3,
        depths_mask_range = None,
        depth_shape_db=(512, 512),
        load_target_depth=False,
        load_cond_depth=False,
        return_depth_input=False,
        o_depth_tform=None,
        depth_scale=1000.0,
        # layout args
        layout_shape_db=(512, 512),
        layout_dir: str = "",
        max_num_points=20,
        intersection_pos="height",
        step=1,
        depth_data_dir=None,
        enable_remapping_cls=False,
        layout_label_ids=None,
        n_scenes=None,
        obj_layout_dir=None,
        obj_layout_range=3.0,
        obj_layout_n_max=20,
        **kwargs,
    ):
        super(Front3DPoseObjLayout, self).__init__(
            image_strides=image_strides,
            graph_dir=graph_dir,
            root_dir=root_dir,
            scene_ids=scene_ids,
            pose_dir=pose_dir,
            image_transforms=image_transforms,
            image_size=image_size,
            mode=mode,
            cond_image_strides=cond_image_strides,
            cond_image_size=cond_image_size,
            target_dir=target_dir,
            n_cond_frames=n_cond_frames,
            n_target_frames=n_target_frames,
            depths_mask_range=depths_mask_range,
            depth_shape_db=depth_shape_db,
            load_target_depth=load_target_depth,
            load_cond_depth=load_cond_depth,
            return_depth_input=return_depth_input,
            o_depth_tform=o_depth_tform,
            depth_scale=depth_scale,
            layout_shape_db=layout_shape_db,
            layout_dir=layout_dir,
            max_num_points=max_num_points,
            intersection_pos=intersection_pos,
            step=step,
            depth_data_dir=depth_data_dir,
            enable_remapping_cls=enable_remapping_cls,
            layout_label_ids=layout_label_ids,
            n_scenes=n_scenes,
            **kwargs,
        )
        self.obj_layout_dir = obj_layout_dir
        self.obj_layout_range = obj_layout_range
        self.obj_layout_n_max = obj_layout_n_max

    def get_layout(
        self,
        poses,
        scene_id: str = None,
        frame_ids: List[str] = None,
        layout_dir: str = None,
        layout_depths: List[torch.FloatTensor] = None,
        layout_clss: List[torch.LongTensor] = None,
    ) -> Dict:
        out = {}
        out.update(
            get_wall_layout(
                self, poses, scene_id, frame_ids, layout_dir, layout_depths, layout_clss
            )
        )
        layout_file = os.path.join(self.obj_layout_dir, f"{scene_id}.json")
        obj_layouts = json.load(open(layout_file, "r"))["boxes"]
        out.update(
            self.get_obj_layout(
                poses, obj_layouts, self.obj_layout_range, self.obj_layout_n_max
            )
        )
        return out
