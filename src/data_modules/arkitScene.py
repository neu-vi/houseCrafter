import os
import math
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import sys
import json
from .threedod.benchmark_scripts.utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt
from .threedod.benchmark_scripts.utils.taxonomy import class_names, ARKitDatasetConfig
import random
from typing import List, Dict, Set, Tuple
from cfg_util import get_obj_from_str, instantiate_from_config
from .data_utils import (
    resize_img,
    get_ray_direction,
    get_plucker_coordinate,
    get_world_pcd,
    recursive_local_expand,
    collate_fn,
    get_connected_subgraphs,
)
from einops import rearrange, repeat
import cv2
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import transforms


class ArkitSceneDataLoader:
    def __init__(
        self,
        root_dir,
        batch_size,
        T_in=3,
        T_out=6,
        num_workers=4,
        depth_tform_cfg=None,
        out_dir = '/projects/vig/hieu/gen_arkit/',
        scene_index = 0
    ):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.T_in = T_in
        self.T_out = T_out
        self.num_workers = num_workers
        self.depth_tform_cfg = depth_tform_cfg
        self.out_dir = out_dir
        self.scene_index = scene_index

        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def train_dataloader(self):
        dataset = ArkitSceneData(
            root_dir=self.root_dir,
            image_transforms=self.image_transforms,
            validation=False,
            T_in=self.T_in,
            T_out=self.T_out,
            depth_tform_cfg=self.depth_tform_cfg,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        dataset = ArkitSceneData(
            root_dir=self.root_dir,
            image_transforms=self.image_transforms,
            validation=True,
            fix_sample = False,
            T_in=self.T_in,
            T_out=self.T_out,
            depth_tform_cfg=self.depth_tform_cfg,
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
    def train_log_dataloader(self):
        dataset = ArkitSceneData(
            root_dir=self.root_dir,
            image_transforms=self.image_transforms,
            validation=False,
            fix_sample = False,
            T_in=self.T_in,
            T_out=self.T_out,
            depth_tform_cfg=self.depth_tform_cfg,
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
    def generative_dataloader(self):
        dataset = ArkitSceneData(
            root_dir=self.root_dir,
            image_transforms=self.image_transforms,
            validation=True,
            fix_sample = True,
            T_in=self.T_in,
            T_out=self.T_out,
            depth_tform_cfg=self.depth_tform_cfg,
            generative_mode=True,
            out_dir=self.out_dir,
            generative_scene_index=self.scene_index,
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn,
        )


def get_pose(transformation):
    # transformation: 4x4
    return transformation


class ArkitSceneData(Dataset):
    def __init__(
        self,
        root_dir="/work/vig/Datasets/ARKITScenes/3dod/Training",
        image_transforms=None,
        # total_view=12,
        validation=False,
        T_in=3,
        T_out=6,
        fix_sample=False,
        depth_tform_cfg: Dict = None,
        generative_mode = False,
        out_dir = None,
        generative_scene_index = 0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.T_in = T_in
        self.T_out = T_out
        self.total_veiw = self.T_in + self.T_out
        self.fix_sample = fix_sample
        self.validation = validation
        self.scene_list = os.path.join(
            self.root_dir, "processed_scene_new.json"
        )
        self.image_strides = [8,16,32,64]
        self.cond_image_strides = [32]
        self.max_num_points = 10
        self.generative_mode = generative_mode
        self.out_dir = out_dir
        self.generative_scene_index = 8
            
        # load scene list
        with open(self.scene_list, "r") as f:
            self.scene_list = json.load(f)
        self.invalid_scene_path = os.path.join(
            self.root_dir, "invalid_scene_record_new.json"
        )
        with open(self.invalid_scene_path, "r") as f:
            self.invalid_scene_list = json.load(f)

        # remove invalid scene
        all_scene_name = list(self.scene_list.keys())
        
        self.paths = []
        for i in range(len(all_scene_name)):
            scene_name = all_scene_name[i]
            if scene_name in self.invalid_scene_list:
                print("invalid scene: ", scene_name)
                continue
            scene_path = os.path.join(self.root_dir, scene_name, f"{scene_name}_frames")
            if not os.path.exists(scene_path):
                print("scene path not exist: ", scene_path)
                continue
            self.paths.append(scene_path)

        total_objects = len(self.paths)
        
        if validation:
            self.paths = self.paths[
                math.floor(total_objects / 100.0 * 90.0) :
            ]  # used last 1% as validation
        else:
            self.paths = self.paths[
                : math.floor(total_objects / 100.0 * 90.0)
            ]  # used first 99% as training
        print('validation: ', validation)
        print("============= length of dataset %d =============" % len(self.paths))
        if generative_mode:
            self.prepare_generative_scene(self.generative_scene_index)
        self.tform = image_transforms
        # todo: dataloader will be used to the specific scenes
        # self.loader = TenFpsDataLoader
        self.o_depth_tform = transforms.Compose(
            [
                lambda x: torch.tensor(x, dtype=torch.float32),
                instantiate_from_config(depth_tform_cfg),
            ]
        )
        self.image_size = 256
        self.max_num_points = 10
        # downscale = 512 / 256.0
        # self.fx = 560.0 / downscale
        # self.fy = 560.0 / downscale
        # self.intrinsic = torch.tensor(
        #     [[self.fx, 0, 128.0, 0, self.fy, 128.0, 0, 0, 1.0]], dtype=torch.float64
        # ).view(3, 3)

    def __len__(self):
        return len(self.paths)

    def get_pose(self, transformation):
        # transformation: 4x4
        return transformation
    
    def prepare_generative_scene(self, scene_index):
        """
        Prepare the scene for generative mode.
        """
        # todo: create a new folder to store the generated images and depths
        scene_name = self.paths[scene_index].split("/")[-2]
        scene_path = os.path.join(self.out_dir, scene_name)
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        self.image_folder = os.path.join(self.out_dir, scene_name, "images")
        self.depth_folder = os.path.join(self.out_dir, scene_name, "depths")
        self.pose_folder = os.path.join(self.out_dir, scene_name, "pose")
        if not os.path.exists(self.pose_folder):
            os.makedirs(self.pose_folder)
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        if not os.path.exists(self.depth_folder):
            os.makedirs(self.depth_folder)
        # Done: make an unvisited list containing the frames to be generated
        loader = TenFpsDataLoader(
            dataset_cfg=None,
            class_names=class_names,
            root_path=self.paths[self.generative_scene_index],
        )
        total_view = self.total_veiw
        stride = 5  # fixed stride between frames

        # Ensure enough frames
        max_start = len(loader) - (total_view - 1) * stride
        if max_start <= 0:
            raise ValueError(f"Scene {self.paths[self.generative_scene_index]} does not have enough frames.")

        # Possible start indices: 0, 5, 10, ...
        frame_list = list(range(0, len(loader), stride))
        # Done: go through the frames frame list first and save the pose in advance:
        for i in range(len(frame_list)):
            frame = loader[frame_list[i]]
            # save the pose
            pose = frame["pose"]
            np.save(os.path.join(self.pose_folder, f"{frame_list[i]:04d}.npy"), pose)
            # save empty image and depth
            image = frame["image"]
            depth = frame["depth"]
            image = np.zeros_like(image)
            depth = np.zeros_like(depth)
            image = Image.fromarray(np.uint8(image[:, :, :3] * 255.0))
            image.save(os.path.join(self.image_folder, f"{frame_list[i]:04d}.png"))
            np.save(os.path.join(self.depth_folder, f"{frame_list[i]:04d}.npy"), depth)
        K_og = frame["intrinsics"]
        # ?: plan the order of frame traversal
        # ? first, choose n frames as the starting points(let it be 3), generate the first 10
        tranversal_order = []
        potential_start = random.sample(frame_list[:10+3], 3)
        initial_gen_frame = frame_list[:10+3]
        # frame_list[:10+3].copy()-potential_start
        visited = set(potential_start)
        tranversal_order.append({
            "cond": potential_start,
            "target": initial_gen_frame,
        })
        visited.update(initial_gen_frame)
        # ?Step 3: Visit the rest of the frames in order
        i = len(visited)
        while i < len(frame_list):
            # Get the next T_out target frames that are not visited
            target = []
            for f in frame_list[i:]:
                if f not in visited:
                    target.append(f)
                if len(target) == self.T_out:
                    break
            if not target:
                break


            # Condition: use the most recent T_in visited frames in order
            cond = sorted(visited)
            cond = cond[-self.T_in:] if len(cond) >= self.T_in else cond
            visited.update(target)

            tranversal_order.append({
                "cond": cond,
                "target": target,
            })

            i += len(target)

        self.tranversal_order = tranversal_order
        self.generative_scene_path = self.paths[self.generative_scene_index]
        self.generative_scene_name = scene_name
        self.paths = self.tranversal_order
        print("tranversal order: ", self.tranversal_order)
        print(f"need {len(self.tranversal_order)} batch to generate")
        


    def load_im(self, path, color):
        """
        replace background pixel with random color in rendering
        """
        try:
            img = plt.imread(path)
        except:
            print("load image failed: ", path)
            # print(path)
            sys.exit()
        # img[img[:, :, -1] == 0.0] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.0))
        return img
    
    def preprocess_loaded_img(self, im):
        # img, K = crop_img(img, K)
        im = im.convert("RGB")
        return self.tform(im)

    def __getitem__(self, index):
        print('doing get item for total length: ', len(self.paths))
        print("index: ", index)
            # loader = TenFpsDataLoader(
        root_path = self.paths[index] if not self.generative_mode else self.generative_scene_path
        loader = TenFpsDataLoader(
            dataset_cfg=None,
            class_names=class_names,
            root_path=root_path,
            )
        data = {}

        if not self.generative_mode:
            total_view = self.total_veiw
            stride = 5  # fixed stride between frames

            # Ensure enough frames
            max_start = len(loader) - (total_view - 1) * stride
            if max_start <= 0:
                raise ValueError(f"Scene {self.paths[index]} does not have enough frames.")

            # Possible start indices: 0, 5, 10, ...ß
            valid_starts = list(range(0, max_start, stride))
            start_index = valid_starts[0] if self.fix_sample else random.choice(valid_starts)

            selected_indices = [start_index + i * stride for i in range(total_view)]
            # randomly select 3 frames among selected indices
            index_inputs = random.sample(selected_indices, self.T_in)
            # the rest are target frames
            index_targets = list(set(selected_indices) - set(index_inputs))
        else:
            index_inputs = self.paths[index]["cond"]
            index_targets = self.paths[index]["target"]
        
        # index_inputs = selected_indices[:self.T_in]
        # index_targets = selected_indices[self.T_in:]

        color = [1.0, 1.0, 1.0, 1.0]
        input_ims, target_ims = [], []
        cond_Ts, target_Ts = [], []

        data = {}
        input_ims, target_ims = [], []
        depth_inputs, depth_targets = [], []
        cond_Ts, cond_Ts_inv = [], []
        target_Ts, target_Ts_inv = [], []

        # Load conditioning (input) frames
        meta = {}
        meta["index_inputs"] = index_inputs
        meta["index_targets"] = index_targets
        meta['image_folder'] = self.image_folder
        meta['depth_folder'] = self.depth_folder
        print("index_inputs: ", index_inputs)
        print("index_targets: ", index_targets)
        for idx in index_inputs:
            frame = loader[idx]

            # Image and depth
            if True:
            # if not self.generative_mode or index == 0:
                im_tensor, K = self.process_im(frame["image"], frame["intrinsics"])
                # if not self.generative_mode or index == 0:
                input_ims.append(im_tensor)
                # K_inputs.append(K)
                if "depth" in frame:
                    depth = frame["depth"].astype(np.float32) / 1000.0
                    depth_tensor = self.preprocess_output_depth(depth)
                    # if not self.generative_mode or index == 0:
                    depth_inputs.append(depth_tensor)
                if not os.path.exists(os.path.join(self.out_dir, self.generative_scene_name, f"K.npy")):
                    np.save(os.path.join(self.out_dir, self.generative_scene_name, f"K.npy"), K)

                # if self.generative_mode:
                #     im = frame["image"]
                #     H, W = im.shape[:2]
                #     # swap the green and blue channels
                #     im = im[:, :, [2, 1, 0]]
                #     im = Image.fromarray(np.uint8(im[:, :, :3]))
                #     im = im.convert("RGB")

                #     crop_size = min(H, W)
                #     top = (H - crop_size) // 2
                #     left = (W - crop_size) // 2
                #     im_cropped = im.crop((left, top, left + crop_size, top + crop_size))

                #     # Resize image
                #     im_resized = im_cropped.resize((self.image_size, self.image_size), Image.BILINEAR)
                #     im_resized.save(os.path.join(self.image_folder, f"{idx:04d}_og.png"))
                #     # save the image and depth to the generated folder
                #     # depth_tensor = depth_tensor.cpu().numpy()
                #     # depth_tensor = depth_tensor.astype(np.float32)*0.001
                #     h, w = depth.shape
                #     min_side = min(h, w)
                #     top = (h - min_side) // 2
                #     left = (w - min_side) // 2
                #     depth_cropped = depth[top:top + min_side, left:left + min_side]

                #     depth_resized = np.array(Image.fromarray(depth_cropped).resize(
                #         (self.image_size, self.image_size), resample=Image.NEAREST))

                #     np.save(os.path.join(self.depth_folder, f"{idx:04d}_og.npy"), depth_resized)
                #     # and save the intrinsic at the base folder if not exist
            # if self.generative_mode and index!=0:
            #     # todo: load the image and depth from the generated folder
            #     im_tensor = self.load_im(
            #         os.path.join(self.out_dir, self.generative_scene_name, "images", f"{idx:04d}.png"), color
            #     )
            #     im_tensor = self.preprocess_loaded_img(im_tensor)
            #     input_ims.append(im_tensor)
            #     depth_tensor = np.load(
            #         os.path.join(self.out_dir, self.generative_scene_name, "depths", f"{idx:04d}.npy")
            #     )
            #     depth_tensor = self.preprocess_output_depth(depth_tensor)
            #     depth_inputs.append(depth_tensor)

            # Pose
            cam_pose = frame["pose"]  # [4, 4] world-to-camera
            cond_Ts.append(cam_pose)
            cond_Ts_inv.append(np.linalg.inv(cam_pose).T)  # (T^-1)^T

        # Load target frames
        for idx in index_targets:
            frame = loader[idx]

            # Image and depth
            im_tensor, K = self.process_im(frame["image"], frame["intrinsics"])
            self.K = K
            target_ims.append(im_tensor)
            # K_targets.append(K)
            if "depth" in frame:
                depth = frame["depth"].astype(np.float32) / 1000.0
                depth_tensor = self.preprocess_output_depth(depth)
                depth_targets.append(depth_tensor)

            # Pose
            cam_pose = frame["pose"]  # [4, 4] world-to-camera
            target_Ts.append(cam_pose)
            target_Ts_inv.append(np.linalg.inv(cam_pose).T)  # (T^-1)^T
            if self.generative_mode:
                im = frame["image"]
                H, W = im.shape[:2]
                # swap the green and blue channels
                im = im[:, :, [2, 1, 0]]
                im = Image.fromarray(np.uint8(im[:, :, :3]))
                im = im.convert("RGB")
                crop_size = min(H, W)
                top = (H - crop_size) // 2
                left = (W - crop_size) // 2
                im_cropped = im.crop((left, top, left + crop_size, top + crop_size))
                # Resize image
                im_resized = im_cropped.resize((self.image_size, self.image_size), Image.BILINEAR)
                im_resized.save(os.path.join(self.image_folder, f"{idx:04d}_og.png"))
                # save the image and depth to the generated folder
                # depth_tensor = depth_tensor.cpu().numpy()
                # depth_tensor = depth_tensor.astype(np.float32)*0.001
                h, w = depth.shape
                min_side = min(h, w)
                top = (h - min_side) // 2
                left = (w - min_side) // 2
                depth_cropped = depth[top:top + min_side, left:left + min_side]
                depth_resized = np.array(Image.fromarray(depth_cropped).resize(
                    (self.image_size, self.image_size), resample=Image.NEAREST))
                np.save(os.path.join(self.depth_folder, f"{idx:04d}_og.npy"), depth_resized)
                # and save the intrinsic at the base folder if not exist

        # Convert poses to tensors
        cond_Ts = torch.from_numpy(np.stack(cond_Ts)).clone()       # [num_cond, 4, 4]
        cond_Ts_inv = torch.from_numpy(np.stack(cond_Ts_inv)).clone() 
        target_Ts = torch.from_numpy(np.stack(target_Ts)).clone() 
        target_Ts_inv = torch.from_numpy(np.stack(target_Ts_inv)).clone() 


        # After conditioning depths are collected
        #? prepare input_pos3d for DeCaPE
        self.cond_image_size = 224
        if len(depth_inputs) > 0:
            # Resize depth to conditioning image size if needed
            resized_cond_depths = [
                resize_img(d.numpy(), self.cond_image_size, interpolation=cv2.INTER_NEAREST)
                for d in depth_inputs
            ]
            resized_cond_depths = [torch.tensor(d, dtype=torch.float32) for d in resized_cond_depths]

            # Assume same intrinsics across frames (from first cond frame)
            I = torch.eye(4, device=resized_cond_depths[0].device)
            c_stride = self.cond_image_strides[0]
            # K = torch.from_numpy(K).float().to(resized_cond_depths[0].device)
            # self.K = K
            self.COND_K = self.K * self.cond_image_size / self.image_size
            self.COND_K[2, 2] = 1.0
            self.COND_K = torch.tensor(self.COND_K, dtype=target_Ts.dtype, device = target_Ts.device)

            pcd = [
                get_world_pcd(I, self.COND_K, d[None, ...], c_stride)
                for d in resized_cond_depths
            ]
            # Rearrange and assign
            in_pos3d = rearrange(pcd, "t 1 h w c -> t h w c")
            # print("in_pos3d shape: ", data["in_pos3d"].shape)

        # Select a base pose (typically first conditioning frame)
        base_pose = cond_Ts[0]
        base_pose_inv = torch.linalg.inv(base_pose)

        self.K = torch.tensor(self.K, dtype=target_Ts.dtype, device = target_Ts.device)
        # base_pose = torch.tensor(base_pose, dtype=torch.float32, device = base_pose.device)
        # cond_Ts = [torch.tensor(pose, dtype=torch.float32, device = base_pose.device) for pose in cond_Ts]
        # target_Ts = [torch.tensor(pose, dtype=torch.float32, device = base_pose.device) for pose in target_Ts]


        # self.COND_K = self.COND_K.float()  # Ensure K is float32
        # base_pose = base_pose.float()
        # cond_Ts = [pose.float() for pose in cond_Ts]
        # self.K = self.K.float()  # Ensure K is float32
        # target_Ts = [pose.float() for pose in target_Ts]

        # Compute cond rays
        # ? prepare rays for direction encoding
        cond_rays = {f"ray_{stride}": [] for stride in self.cond_image_strides}
        H = W = self.cond_image_size
        for pose in cond_Ts:
            P_rel = base_pose @ torch.linalg.inv(pose)
            # R = P_rel[:3, :3]
            R = P_rel[:3, :3]

            for stride in self.cond_image_strides:
                ray = get_ray_direction(R, self.COND_K, H, W, stride)
                plucker = get_plucker_coordinate(ray, P_rel[:3, 3])
                cond_rays[f"ray_{stride}"].append(plucker)

        # Compute target rays
        target_rays = {f"ray_{stride}": [] for stride in self.image_strides}
        H = W = self.image_size
        for pose in target_Ts:
            P_rel = base_pose @ torch.linalg.inv(pose)
            R = P_rel[:3, :3]
            for stride in self.image_strides:
                ray = get_ray_direction(R, self.K, H, W, stride)
                plucker = get_plucker_coordinate(ray, P_rel[:3, 3])
                target_rays[f"ray_{stride}"].append(plucker)

        # Assemble final data dict
        target_Ts_inv = rearrange(torch.linalg.inv(target_Ts), "b c d -> b d c")
        cond_Ts_inv = rearrange(torch.linalg.inv(cond_Ts), "b c d -> b d c")

        #? prepare input and target images and depths
        data = {
            "image_input": torch.stack(input_ims),               # [num_cond, 3, H, W]
            "image_target": torch.stack(target_ims),             # [num_target, 3, H, W]
            "pose_in": cond_Ts,                                  # [num_cond, 4, 4]
            "pose_in_inv": cond_Ts_inv,
            "pose_out": target_Ts,
            "pose_out_inv": target_Ts_inv,
            "in_pos3d": in_pos3d,
            'meta': meta,
        }

        if len(depth_inputs) > 0:
            data["depth_input"] = torch.from_numpy(np.stack(depth_inputs)).clone() 
        if len(depth_targets) > 0:
            data["depth_target"] = torch.from_numpy(np.stack(depth_targets)).clone() 

        for k, v in cond_rays.items():
            data[f"cond_{k}"] = rearrange(v, "t h w c -> t c h w")
            # print(f"cond_{k} shape: {data[f'cond_{k}'].shape}")

        for k, v in target_rays.items():
            data[f"target_{k}"] = rearrange(v, "t h w c -> t c h w")
            # print(f"target_{k} shape: {data[f'target_{k}'].shape}")

        self.LAYOUT_K = self.K / min(self.image_strides)
        self.LAYOUT_K[2, 2] = 1.0
        # print('image_inputs shape: ', data["image_input"].shape)
        # print('image_targets shape: ', data["image_target"].shape)
        # print('depth_inputs shape: ', data["depth_input"].shape if "depth_input" in data else None)
        # print('depth_targets shape: ', data["depth_target"].shape if "depth_target" in data else None)
        # print('pose_in shape: ', data["pose_in"].shape)
        # print('pose_out shape: ', data["pose_out"].shape)
        # print('pose_in_inv shape: ', data["pose_in_inv"].shape)
        # print('pose_out_inv shape: ', data["pose_out_inv"].shape)
        # print('batch keys: ', data.keys())
        # print('inpos3d shape: ', data["in_pos3d"].shape if "in_pos3d" in data else None)

        data.update(
            self.get_layout(
                poses=target_Ts,
                scene_id=root_path.split("/")[-2],
                frame_ids=index_targets,
            )
        )
        # print('depth input before passing', data["depth_input"].shape if "depth_input" in data else None)
        # print('in_pos3d shape: ', data["in_pos3d"].shape if "in_pos3d" in data else None)
        return data

    def get_layout(
        self,
        poses: torch.FloatTensor,
        scene_id: str,
        frame_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Load layout cls and pos from file-based storage.

        Args:
            poses: [T, 4, 4] world-to-camera poses (w2c)
            scene_id: str, scene folder name
            frame_ids: List[int], frame ids corresponding to poses

        Returns:
            Dict with padded layout_cls [T, N, H, W] and layout_pos [T, C, H, W]
        """
        layout = {"layout_cls": [], "layout_pos": []}
        layout_base = os.path.join(self.root_dir, scene_id, "layout_per_frames_finetune")
        final_size = self.image_size
        layout_size = self.image_size // min(self.image_strides)

        for i, (pose, frame_id) in enumerate(zip(poses, frame_ids)):
            frame_id_str = str(frame_id).zfill(4)
            label_path = os.path.join(layout_base, str(frame_id), f"layout_label_{frame_id_str}.npy")
            pos_path = os.path.join(layout_base, str(frame_id), f"layout_pos_{frame_id_str}.npy")
            

            # print('loading layout from: ', label_path)
            if not os.path.isfile(label_path) or not os.path.isfile(pos_path):
                raise FileNotFoundError(f"Missing layout files for {label_path} or {pos_path}")

            # Load and convert to tensors
            cls = torch.from_numpy(np.load(label_path)).clone().long()   # [N, H, W]
            pos = torch.from_numpy(np.load(pos_path)).clone().float()*0.001    # [N, H, W]

            # Center crop to square
            _, H, W = cls.shape
            crop_size = min(H, W)
            top = (H - crop_size) // 2
            left = (W - crop_size) // 2
            cls = cls[:, top:top+crop_size, left:left+crop_size]
            pos = pos[:, top:top+crop_size, left:left+crop_size]

            # Resize
            layout_cls = F.interpolate(cls.unsqueeze(0).float(), size=layout_size, mode='nearest').squeeze(0).long()
            depth = F.interpolate(pos.unsqueeze(0), size=layout_size, mode='nearest').squeeze(0)
            
            if len(layout_cls) > self.max_num_points:
                # split wall and furniture, only do cutdown on furniture
                w_cls = layout_cls[:1]
                w_depth = depth[:1]

                f_cls = layout_cls[1:]
                f_depth = depth[1:]

                # todo: figure out if this mask is correct
                mask = f_depth < 0.1  # min depth is 0.1m
                f_depth[mask] = 10000.0 # infinitely far
                indices = torch.argsort(f_depth, dim=0)  # n,h,w
                f_cls = torch.take_along_dim(f_cls, indices, dim=0)
                f_depth = torch.take_along_dim(f_depth, indices, dim=0)
                mask = torch.take_along_dim(mask, indices, dim=0)
                f_depth[mask] = 0.0
                f_depth = f_depth[: self.max_num_points - 1]
                f_cls = f_cls[: self.max_num_points - 1]

                layout_cls = torch.cat([w_cls, f_cls], dim=0)
                depth = torch.cat([w_depth, f_depth], dim=0)

            layout["layout_cls"].append(layout_cls)
            c2w = torch.linalg.inv(pose)
            pos_3d = get_world_pcd(c2w, self.LAYOUT_K, depth)
            layout["layout_pos"].append(torch.stack([depth, pos_3d[..., 1]], dim=1))

            # print(f"layout_cls shape: {layout_cls.shape}, layout_pos shape: {depth.shape}")

        layout["layout_cls"] = pad_sequence(layout["layout_cls"], batch_first=True)  # [T, N, H, W]
        layout["layout_pos"] = pad_sequence(layout["layout_pos"], batch_first=True)  # [T, N, 2, H, W]

        return layout

    def process_im(self, im, K):
        """
        Process image:
        - Center-crop to square
        - Resize to image_size
        - Adjust intrinsic matrix K accordingly
        """
        H, W = im.shape[:2]
        # swap the green and blue channels
        im = im[:, :, [2, 1, 0]]
        im = Image.fromarray(np.uint8(im[:, :, :3]))
        im = im.convert("RGB")

        crop_size = min(H, W)
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        im_cropped = im.crop((left, top, left + crop_size, top + crop_size))

        # Resize image
        im_resized = im_cropped.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Adjust intrinsics
        K = K.copy()
        K[0, 2] -= left
        K[1, 2] -= top
        scale = self.image_size / crop_size
        K[:2] *= scale

        im_tensor = self.tform(im_resized)
        return im_tensor, K

    def preprocess_output_depth(self, depth):
        """
        Preprocess ARKitScenes depth: crop to square, resize to image_size, convert to tensor.
        """
        h, w = depth.shape
        min_side = min(h, w)
        top = (h - min_side) // 2
        left = (w - min_side) // 2
        depth_cropped = depth[top:top + min_side, left:left + min_side]

        depth_resized = np.array(Image.fromarray(depth_cropped).resize(
            (self.image_size, self.image_size), resample=Image.NEAREST))

        # Apply transform
        return self.o_depth_tform(depth_resized)
