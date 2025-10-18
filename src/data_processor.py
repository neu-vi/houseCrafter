import numpy as np
import torch
import torch.nn.functional as F
from data_modules.data_utils import (
    collate_fn,
    get_plucker_coordinate,
    get_ray_direction,
    get_world_pcd,
)
from data_modules.layout_renderer import TorchLayoutRenderer
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


class DataProcessor:
    """
    mostly copy from front3d_dataset
    """

    FRONT3D_BASE_MATRIX = torch.tensor(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=torch.float32,
    )

    FOV = 90

    def __init__(
        self,
        model_cfg,
        device,
        layout_file,
        label_id_mapping_file,
        house_file,
        filter_mesh_class=False,
        skip_layout_class=set()
    ):
        cfg = model_cfg.data.params
        self.img_size = cfg.image_size
        self.cond_img_size = cfg.get("cond_image_size", 224)
        self.img_strides = cfg.image_strides
        self.cond_img_strides = cfg.cond_image_strides
        self._set_K(self.img_size, self.cond_img_size, self.img_strides)

        self.load_cond_depth = cfg.load_cond_depth
        self.layout_size = self.img_size // min(self.img_strides)
        self.layout_renderer = TorchLayoutRenderer(
            device, 512, filter_mesh_class=filter_mesh_class
        )
        self.layout_renderer.set_house_mesh(house_file, label_id_mapping_file)
        self.layout_renderer.set_layout_mesh(layout_file, skip_ids=skip_layout_class)
        self.intersection_pos = cfg.get("intersection_pos", "height")  ############

        # self.intersection_pos = cfg.get("intersection_pos", "3d")

    def make_data_item(self, input_poses, input_rgbs, input_depths, target_poses):
        """
        return data in the format as dataset
        given the config

        input_rgbs: n,h,w,3 np uint8
        input_depth: n,h,w np float32 in meter
        input_poses: n,4,4 np c2w matrix
        target_poses: m,4,4 np c2w matrix
        """
        assert len(input_rgbs) == len(input_depths) == len(input_poses)
        assert input_depths.shape[1] == self.img_size

        input_rgbs = self._process_img(input_rgbs)
        input_depths = torch.tensor(input_depths, dtype=torch.float32)
        input_poses = torch.tensor(input_poses)
        w2c_in = torch.linalg.inv(input_poses)
        target_poses = torch.tensor(target_poses)
        w2c_out = torch.linalg.inv(target_poses)

        # make cond ray
        cond_pose = w2c_in[0]

        cond_rays = {f"ray_{stride}": [] for stride in self.cond_img_strides}
        H = W = self.cond_img_size
        for pose in w2c_in:
            # current cam to selected cam
            P_rel = cond_pose @ torch.linalg.inv(pose)
            # transform cam in get_ray_direction to 3dfront cam first
            R = P_rel[:3, :3] @ self.FRONT3D_BASE_MATRIX
            for stride in self.cond_img_strides:
                ray = get_ray_direction(R, self.COND_K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                cond_rays[f"ray_{stride}"].append(flucker_coords)

        # make target ray
        target_rays = {f"ray_{stride}": [] for stride in self.img_strides}
        H = W = self.img_size
        for pose in w2c_out:
            # current cam to selected cam
            P_rel = cond_pose @ torch.linalg.inv(pose)
            # transform cam in get_ray_direction to 3dfront cam first
            R = P_rel[:3, :3] @ self.FRONT3D_BASE_MATRIX
            for stride in self.img_strides:
                ray = get_ray_direction(R, self.K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                target_rays[f"ray_{stride}"].append(flucker_coords)

        data = {}
        data["image_input"] = input_rgbs
        data["depth_input"] = input_depths
        data["pose_out"] = w2c_out
        data["pose_out_inv"] = rearrange(torch.linalg.inv(w2c_out), "b c d -> b d c")
        data["pose_in"] = w2c_in
        data["pose_in_inv"] = rearrange(torch.linalg.inv(w2c_in), "b c d -> b d c")
        for k, v in target_rays.items():
            data[f"target_{k}"] = rearrange(v, "t h w c -> t c h w")

        for k, v in cond_rays.items():
            data[f"cond_{k}"] = rearrange(v, "t h w c -> t c h w")

        # make pcd
        if self.load_cond_depth:
            input_depths = F.interpolate(
                input_depths.unsqueeze(1), size=self.cond_img_size, mode="nearest"
            ).squeeze(1)
            I = torch.eye(4)
            assert len(self.cond_img_strides) == 1
            c_stride = self.cond_img_strides[0]
            pcd = get_world_pcd(I, self.COND_K, input_depths, c_stride)
            data["in_pos3d"] = pcd

        # get layout
        layout_cls, layout_pos = [], []
        for pose in target_poses:
            depth, clss = self.layout_renderer.render_layout(pose)
            depth = depth.cpu()
            clss = clss.cpu()
            if depth.size(1) != self.layout_size:
                depth = F.interpolate(
                    depth[:, None], size=self.layout_size, mode="nearest"
                ).squeeze(1)
                clss = (
                    F.interpolate(
                        clss[:, None].float(),
                        size=self.layout_size,
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                )
            layout_cls.append(clss)

            R = pose[:3, :3] @ self.FRONT3D_BASE_MATRIX
            pose = pose.clone()
            pose[:3, :3] = R
            pos_3d = get_world_pcd(pose, self.LAYOUT_K, depth)
            if self.intersection_pos == "height":
                layout_pos.append(torch.stack([depth, pos_3d[..., 1]], dim=1))
            elif self.intersection_pos == "3d":
                depth = rearrange(depth, "n h w -> n 1 h w")
                pos_3d = rearrange(pos_3d, "n h w c -> n c h w")
                layout_pos.append(torch.cat([depth, pos_3d], dim=1))

        data["layout_cls"] = pad_sequence(layout_cls, batch_first=True)
        data["layout_pos"] = pad_sequence(layout_pos, batch_first=True)

        return collate_fn([data])

    def _process_img(self, img):
        """
        img: n,h,w,3 np uint8
        return torch tensor n,3,h,w in range (-1,1)
        """
        img = img.astype(np.float32) / 127.5 - 1
        img = rearrange(img, "n h w c -> n c h w")
        img = torch.tensor(img)
        return img

    def _set_K(self, image_size, cond_image_size, image_strides):
        # compute intrinsic matrix
        center = image_size / 2
        focal = image_size / 2 / np.tan(np.radians(self.FOV / 2))
        self.K = torch.tensor(
            [
                [focal, 0.0, center],
                [0.0, focal, center],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float,
        )

        self.COND_K = self.K * cond_image_size / image_size
        self.COND_K[2, 2] = 1.0

        self.LAYOUT_K = self.K / min(image_strides)
        self.LAYOUT_K[2, 2] = 1.0
