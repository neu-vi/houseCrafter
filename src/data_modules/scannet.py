import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# import pyquaternion as qt
from einops import rearrange
import random
from tqdm import tqdm
import json

from .data_utils import crop_img, resize_img, get_ray_direction, get_plucker_coordinate


class ScannetData(Dataset):
    def __init__(
        self,
        scene_num_views,
        root_dir="/work/vig/Datasets/ScanNet/scans_uncomp/",
        image_transforms=[],
        image_size=256,
        mode="train",
        T_in=1,
        T_out=1,
    ) -> None:
        self.root_dir = root_dir
        self.tform = image_transforms
        self.image_size = image_size
        self.mode = mode
        self.T_in = T_in
        self.T_out = T_out
        self.scene_num_views = json.load(open(scene_num_views))
        scenes = os.listdir(root_dir)
        if len(scenes) == 0:
            raise Exception(
                f"No scenes in {root_dir}, please check ScanNet is in the correct path"
            )
        else:
            print("Found {} scenes".format(len(scenes)))

        self.scene_list = []
        print("checking valid scenes")
        for scene_id in tqdm(sorted(scenes)):
            if scene_id not in self.scene_num_views:
                continue
            if self.scene_num_views[scene_id] < 20 * (self.T_in + self.T_out + 1):
                continue

            scene_dir = os.path.join(root_dir, scene_id)
            self.scene_list.append(scene_dir)
        print("Found {} valid scenes".format(len(self.scene_list)))

    def __len__(self):
        return len(self.scene_list)

    def load_frame_data(self, scene_path: str, frame_id: int):
        """
        load img, extrinsic, intrinsic
        and prompt

        """
        img_path = os.path.join(scene_path, "color", f"frame-{frame_id:06}.color.jpg")
        if not os.path.isfile(img_path):
            img_path = os.path.join(scene_path, "color", f"{frame_id}.jpg")
        img = plt.imread(img_path)

        # cam2world matrix
        P_path = os.path.join(scene_path, "pose", f"frame-{frame_id:06}.pose.txt")
        if not os.path.isfile(P_path):
            P_path = os.path.join(scene_path, "pose", f"{frame_id}.txt")
        P = np.loadtxt(P_path)

        # cam2uv matrix
        K_path = os.path.join(scene_path, "intrinsic", "intrinsic_color.txt")
        K = np.loadtxt(K_path).astype(np.float32)[:3, :3]
        return img, P, K

    def __getitem__(self, index):
        idx = index
        data = {}
        scene_root = self.scene_list[idx]
        num_frames = len(os.listdir(os.path.join(scene_root, "color")))

        current_frame_id = 0
        while True:
            if self.mode == "train":
                cond_frame_id_start = random.randint(
                    0, num_frames - 20 * (self.T_in + self.T_out + 1)
                )

            else:
                cond_frame_id_start = current_frame_id

            cond_frame_ids = list(
                range(cond_frame_id_start, cond_frame_id_start + self.T_in * 20, 20)
            )

            target_frame_ids = list(
                range(
                    cond_frame_id_start + (self.T_in) * 20,
                    cond_frame_id_start + (self.T_in + self.T_out) * 20,
                    20,
                )
            )

            # todo: make the frame the starting coordinate
            # cond_frame, P, K = self.load_frame_data(scene_root, cond_frame_id_start)
            # # cond_frame, K = self.preprocess_img(cond_frame, K)
            # P_inv = np.linalg.inv(P)
            # if np.isnan(P_inv).any() or np.isinf(P_inv).any():
            #     # print(
            #     #     f"got singular pose matrix scene {scene_root} frames {cond_frame_id} {P}"
            #     # )
            #     current_frame_id += 1
            #     continue

            Pc = []
            cond_frames = []
            valid_cond = True
            for cond_frame_id in cond_frame_ids:
                frame, P, K = self.load_frame_data(scene_root, cond_frame_id)
                if np.isnan(P).any() or np.isinf(P).any():
                    valid_cond = False
                    break
                frame, K = self.preprocess_img(frame, K)
                cond_frames.append(frame)
                # get world 2 cam matrix
                P = np.linalg.inv(P)
                P = torch.tensor(P).float()
                Pc.append(P)
            if not valid_cond:
                current_frame_id += 1
                continue

            Pt = []
            tg_frames = []
            valid_target = True
            for target_frame_id in target_frame_ids:
                frame, P, K = self.load_frame_data(scene_root, target_frame_id)
                if np.isnan(P).any() or np.isinf(P).any():
                    valid_target = False
                    break
                frame, K = self.preprocess_img(frame, K)
                tg_frames.append(frame)
                # get world 2 cam matrix
                P = np.linalg.inv(P)
                P = torch.tensor(P).float()
                Pt.append(P)
            if not valid_target:
                current_frame_id += 1
                continue

            cond_frames = torch.stack(cond_frames, dim=0)
            tg_frames = torch.stack(tg_frames, dim=0)
            Pc = torch.stack(Pc, dim=0)
            Pt = torch.stack(Pt, dim=0)
            break

        data["image_input"] = cond_frames
        data["image_target"] = tg_frames
        data["pose_out"] = Pt
        data["pose_out_inv"] = rearrange(torch.linalg.inv(Pt), "b c d -> b d c")
        data["pose_in"] = Pc
        data["pose_in_inv"] = rearrange(torch.linalg.inv(Pc), "b c d -> b d c")

        return data

    def preprocess_img(self, img, K):
        img, K = crop_img(img, K)
        img, K = resize_img(img, K, self.image_size)
        # img = Image.fromarray(np.uint8(img[:, :, :3] * 255.0))
        img = Image.fromarray(img)

        img = img.convert("RGB")
        return self.tform(img), K


class RCNScannetData(ScannetData):
    def __init__(
        self,
        scene_num_views,
        root_dir="/work/vig/Datasets/ScanNet/scans_uncomp/",
        image_transforms=[],
        image_size=256,
        mode="train",
        T_in=1,
        T_out=1,
        image_strides=[8, 16, 32, 64],
    ) -> None:
        super(RCNScannetData, self).__init__(
            scene_num_views=scene_num_views,
            root_dir=root_dir,
            image_transforms=image_transforms,
            image_size=image_size,
            mode=mode,
            T_in=T_in,
            T_out=T_out,
        )
        self.image_strides = image_strides

    def __getitem__(self, index):
        idx = index
        data = {}
        scene_root = self.scene_list[idx]
        num_frames = len(os.listdir(os.path.join(scene_root, "color")))

        current_frame_id = 0
        while True:
            if self.mode == "train":
                cond_frame_id_start = random.randint(
                    0, num_frames - 20 * (self.T_in + self.T_out + 1)
                )
            else:
                cond_frame_id_start = current_frame_id

            cond_frame_ids = list(
                range(cond_frame_id_start, cond_frame_id_start + self.T_in * 20, 20)
            )

            target_frame_ids = list(
                range(
                    cond_frame_id_start + (self.T_in) * 20,
                    cond_frame_id_start + (self.T_in + self.T_out) * 20,
                    20,
                )
            )

            Pc = []
            cond_frames = []
            valid_cond = True
            for cond_frame_id in cond_frame_ids:
                frame, P, K = self.load_frame_data(scene_root, cond_frame_id)
                if np.isnan(P).any() or np.isinf(P).any():
                    valid_cond = False
                    break
                frame, K = self.preprocess_img(frame, K)
                cond_frames.append(frame)
                # get world 2 cam matrix
                P = np.linalg.inv(P)
                P = torch.tensor(P).float()
                Pc.append(P)
            if not valid_cond:
                current_frame_id += 1
                continue

            Pt = []
            tg_frames = []
            target_rays = {f"ray_{stride}": [] for stride in self.image_strides}

            # select a condition pose
            frame_id = np.random.randint(self.T_in) if self.mode == "train" else 0
            cond_pose = Pc[frame_id].numpy()  # world2cam

            valid_target = True
            for target_frame_id in target_frame_ids:
                frame, P, K = self.load_frame_data(scene_root, target_frame_id)
                if np.isnan(P).any() or np.isinf(P).any():
                    valid_target = False
                    break
                frame, K = self.preprocess_img(frame, K)
                C, H, W = frame.size()

                tg_frames.append(frame)

                # compute ray
                ## target_cam 2 cond_cam
                P_rel = cond_pose @ P
                P_rel = torch.tensor(P_rel).float()
                for stride in self.image_strides:
                    ray = get_ray_direction(P_rel[:3, :3], K, H, W, stride)
                    flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                    target_rays[f"ray_{stride}"].append(flucker_coords)

                # get world 2 cam matrix
                P = np.linalg.inv(P)
                P = torch.tensor(P).float()
                Pt.append(P)
            if not valid_target:
                current_frame_id += 1
                continue

            cond_frames = torch.stack(cond_frames, dim=0)
            tg_frames = torch.stack(tg_frames, dim=0)
            Pc = torch.stack(Pc, dim=0)
            Pt = torch.stack(Pt, dim=0)
            break

        data["image_input"] = cond_frames
        data["image_target"] = tg_frames
        data["pose_out"] = Pt
        data["pose_out_inv"] = rearrange(torch.linalg.inv(Pt), "b c d -> b d c")
        data["pose_in"] = Pc
        data["pose_in_inv"] = rearrange(torch.linalg.inv(Pc), "b c d -> b d c")
        for k, v in target_rays.items():
            data[f"target_{k}"] = rearrange(v, "t h w c -> t c h w")
        return data
