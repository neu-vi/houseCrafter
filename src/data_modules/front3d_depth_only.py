import os
import zlib
from typing import Dict, List, Set, Tuple

import lmdb
import numpy as np
import torch
from einops import rearrange, repeat

from .data_utils import get_plucker_coordinate, get_ray_direction
from .front3d import FRONT3D_BASE_MATRIX, Front3DPose


class Front3DDepthOnly(Front3DPose):
    """
    dataset for depth generation model
    depth is given as input and target
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
        o_depth_tform=None,
        depth_scale=1000.0,
        # layout args
        layout_shape_db=(512, 512),
        layout_dir: str = "",
        max_num_points=20,
        intersection_pos="height",
        step=1,
        **kwargs,
    ):
        super(Front3DDepthOnly, self).__init__(
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
            o_depth_tform=o_depth_tform,
            depth_scale=depth_scale,
            layout_shape_db=layout_shape_db,
            layout_dir=layout_dir,
            max_num_points=max_num_points,
            intersection_pos=intersection_pos,
            step=step,
        )

    def load_frames_data(self, scene_id, frame_ids: List[str], is_target=False):
        """
        load frames data from lmdb: rgb, pose (w2c), depth (optional)
        """
        if self.scene_paths is None:
            if is_target:
                data_dir = self.target_dir
            else:
                data_dir = self.data_dir
        else:
            data_dir = self.scene_paths[scene_id]["img_path"]

        db = lmdb.open(
            os.path.join(data_dir, scene_id),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        poses, depths = [], []
        with db.begin(write=False) as txn:
            for frame_id in frame_ids:
                # rgb_key = f"{frame_id}_rgb".encode("ascii")
                # img = txn.get(rgb_key)
                # img = np.frombuffer(img, dtype=np.uint8)
                # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                # img = self.preprocess_img(img)
                # rgbs.append(img)

                pose_key = f"{frame_id}_pose".encode("ascii")
                c2w = txn.get(pose_key)
                c2w = (
                    np.frombuffer(zlib.decompress(c2w), dtype=np.float32)
                    .reshape(4, 4)
                    .copy()
                )
                c2w[3, 3] = 1.0  # bug when save pose
                w2c = np.linalg.inv(c2w)
                poses.append(torch.tensor(w2c, dtype=torch.float32))

                depth_key = f"{frame_id}_depth".encode("ascii")
                depth = zlib.decompress(txn.get(depth_key))
                # depth = np.frombuffer(depth, dtype=np.uint16).reshape(
                #     *self.depth_shape_db
                # )
                depth = np.frombuffer(depth, dtype=np.uint16)
                size = int(np.sqrt(len(depth)))
                depth = depth.reshape(size, size)
                # convert from mm to m in this case depth_scale = 1000
                depth = depth.astype(np.float32) / self.depth_scale
                depths.append(depth)
        db.close()

        return poses, depths

    def get_item_from_meta(
        self,
        scene_id: str,
        frames_meta: Dict[str, List[str]],
        load_target_GT=True,
        target_pose=None,
    ):
        """
        args: frames_meta: {
                cond: [view_id],
                target: [view_id],
            }
            target_poses: list of  w2c 4,4 tensor

        return {
            "image_input": shape (num_cond, 3, H, W), *DEPTH IMAGE
            "image_target": shape (num_target, 3, H, W), *DEPTH IMAGE
            "pose_out": shape (num_target, 4, 4) w2c of target frames
            "pose_out_inv": shape (num_target, 4, 4) transpose of inverse of pose_out
            "pose_in": shape (num_cond, 4, 4) w2c of cond frames
            "pose_in_inv": shape (num_cond, 4, 4) transpose of inverse of pose_in
        }
        """
        # load cond frames
        Pc, cond_depths = self.load_frames_data(scene_id, frames_meta["cond"])

        ## select a condition pose
        frame_id = np.random.randint(len(Pc)) if self.mode == "train" else 0
        cond_pose = Pc[frame_id]  # world2cam

        # compute ray for cond images
        cond_rays = {f"ray_{stride}": [] for stride in self.cond_image_strides}
        H = W = self.cond_image_size
        for pose in Pc:
            # current cam to selected cam
            P_rel = cond_pose @ torch.linalg.inv(pose)
            # transform cam in get_ray_direction to 3dfront cam first
            R = P_rel[:3, :3] @ torch.tensor(FRONT3D_BASE_MATRIX, device=P_rel.device)
            for stride in self.cond_image_strides:
                ray = get_ray_direction(R, self.COND_K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                cond_rays[f"ray_{stride}"].append(flucker_coords)

        # load target frames
        if load_target_GT:
            Pt, tg_depths = self.load_frames_data(
                scene_id,
                frames_meta["target"],
                is_target=True,
            )
        else:
            tg_depths = None
            Pt = target_pose
            assert Pt is not None

        # compute ray for target images
        target_rays = {f"ray_{stride}": [] for stride in self.image_strides}
        # C, H, W = tg_frames[0].size()
        # assert C == 3
        H = W = self.image_size

        for pose in Pt:
            # current cam to selected cam
            P_rel = cond_pose @ torch.linalg.inv(pose)
            # transform cam in get_ray_direction to 3dfront cam first
            R = P_rel[:3, :3] @ torch.tensor(FRONT3D_BASE_MATRIX, device=P_rel.device)
            for stride in self.image_strides:
                ray = get_ray_direction(R, self.K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                target_rays[f"ray_{stride}"].append(flucker_coords)

        # cond_frames = torch.stack(cond_frames, dim=0)
        Pc = torch.stack(Pc, dim=0)
        Pt = torch.stack(Pt, dim=0)

        data = {}

        data["pose_out"] = Pt
        data["pose_out_inv"] = rearrange(torch.linalg.inv(Pt), "b c d -> b d c")
        data["pose_in"] = Pc
        data["pose_in_inv"] = rearrange(torch.linalg.inv(Pc), "b c d -> b d c")
        data["_base_pose"] = cond_pose
        data["scene_id"] = scene_id
        data["frame_ids"] = frames_meta

        for k, v in target_rays.items():
            data[f"target_{k}"] = rearrange(v, "t h w c -> t c h w")

        for k, v in cond_rays.items():
            data[f"cond_{k}"] = rearrange(v, "t h w c -> t c h w")

        # process cond depth
        cond_depths = [self.preprocess_output_depth(d) for d in cond_depths]
        cond_depths = repeat(cond_depths, "t h w -> t c h w", c=3)
        data["image_input"] = cond_depths

        # process target depth
        if load_target_GT:
            tg_depths = [self.preprocess_output_depth(d) for d in tg_depths]
            tg_depths = repeat(tg_depths, "t h w -> t c h w", c=3)
            data["image_target"] = tg_depths

        return data
