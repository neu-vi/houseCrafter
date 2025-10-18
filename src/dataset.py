import os
import math
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import sys
import cv2
# import pyquaternion as qt
from einops import rearrange, repeat
from torch import Tensor
from typing import Tuple
import random
from torchvision import transforms
from tqdm import tqdm
import json

class ObjaverseDataLoader():
    def __init__(self, root_dir, batch_size, total_view=12, num_workers=4):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        image_transforms = [torchvision.transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True,
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def get_pose(transformation):
    # transformation: 4x4
    return transformation

class ObjaverseData(Dataset):
    def __init__(self,
                 root_dir='.objaverse/hf-objaverse-v1/views',
                 image_transforms=None,
                 total_view=12,
                 validation=False,
                 T_in=1,
                 T_out=1,
                 fix_sample=False,
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.total_view = total_view
        self.T_in = T_in
        self.T_out = T_out
        self.fix_sample = fix_sample

        self.paths = []
        # # include all folders
        # for folder in os.listdir(self.root_dir):
        #     if os.path.isdir(os.path.join(self.root_dir, folder)):
        #         self.paths.append(folder)
        # load ids from .npy so we have exactly the same ids/order
        self.paths = np.load("../scripts/obj_ids.npy")
        # # only use 100K objects for ablation study
        # self.paths = self.paths[:100000]
        total_objects = len(self.paths)
        assert total_objects == 790152, 'total objects %d' % total_objects
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

        downscale = 512 / 256.
        self.fx = 560. / downscale
        self.fy = 560. / downscale
        self.intrinsic = torch.tensor([[self.fx, 0, 128., 0, self.fy, 128., 0, 0, 1.]], dtype=torch.float64).view(3, 3)

    def __len__(self):
        return len(self.paths)

    def get_pose(self, transformation):
        # transformation: 4x4
        return transformation


    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = 12

        if self.fix_sample:
            if self.T_out > 1:
                indexes = range(total_view)
                index_targets = list(indexes[:2]) + list(indexes[-(self.T_out-2):])
                index_inputs = indexes[1:self.T_in+1]   # one overlap identity
            else:
                indexes = range(total_view)
                index_targets = indexes[:self.T_out]
                index_inputs = indexes[self.T_out-1:self.T_in+self.T_out-1] # one overlap identity
        else:
            assert self.T_in + self.T_out <= total_view
            # training with replace, including identity
            indexes = np.random.choice(range(total_view), self.T_in+self.T_out, replace=True)
            index_inputs = indexes[:self.T_in]
            index_targets = indexes[self.T_in:]
        filename = os.path.join(self.root_dir, self.paths[index])

        color = [1., 1., 1., 1.]

        try:
            input_ims = []
            target_ims = []
            target_Ts = []
            cond_Ts = []
            for i, index_input in enumerate(index_inputs):
                input_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_input), color))
                input_ims.append(input_im)
                input_RT = np.load(os.path.join(filename, '%03d.npy' % index_input))
                cond_Ts.append(self.get_pose(np.concatenate([input_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(np.concatenate([target_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
        except:
            # print('error loading data ', filename)
            filename = os.path.join(self.root_dir, '0a01f314e2864711aa7e33bace4bd8c8')  # this one we know is valid
            input_ims = []
            target_ims = []
            target_Ts = []
            cond_Ts = []
            # very hacky solution, sorry about this
            for i, index_input in enumerate(index_inputs):
                input_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_input), color))
                input_ims.append(input_im)
                input_RT = np.load(os.path.join(filename, '%03d.npy' % index_input))
                cond_Ts.append(self.get_pose(np.concatenate([input_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(np.concatenate([target_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))

        # stack to batch
        data['image_input'] = torch.stack(input_ims, dim=0)
        data['image_target'] = torch.stack(target_ims, dim=0)
        data['pose_out'] = np.stack(target_Ts)
        data['pose_out_inv'] = np.linalg.inv(np.stack(target_Ts)).transpose([0, 2, 1])
        data['pose_in'] = np.stack(cond_Ts)
        data['pose_in_inv'] = np.linalg.inv(np.stack(cond_Ts)).transpose([0, 2, 1])
        # print('image_input', data['image_input'].shape)
        # print('image_target', data['image_target'].shape)
        # print('pose_out', data['pose_out'].shape)
        # print('pose_out_inv', data['pose_out_inv'].shape)
        # print('pose_in', data['pose_in'].shape)
        # print('pose_in_inv', data['pose_in_inv'].shape)
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

class ScannetDataloader():
    def __init__(
        self,
        root_dir,
        batch_size,
        image_size = 256,
        num_views=3,
        num_workers=4,
        val_root_dir=None,
    ):
        self.root_dir = root_dir
        self.val_root_dir = val_root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_views = num_views

        image_transforms = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ]
        self.image_transforms = transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ScannetData(
            root_dir=self.root_dir,
            image_size=self.image_size,
            image_transforms=self.image_transforms,
            num_views=self.num_views,
            mode="train",
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
        )

    def val_dataloader(self):
        if self.val_root_dir is None:
            UserWarning("val root dir is none, using the training root dir")
            root_dir = self.root_dir
        else:
            root_dir = self.val_root_dir
        dataset = ScannetData(
            root_dir=root_dir,
            image_size=self.image_size,
            image_transforms=self.image_transforms,
            num_views=self.num_views,
            mode="val"
        )
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class ScannetData(Dataset):
    def __init__(
        self,
        scene_num_views,
        root_dir="/work/vig/Datasets/ScanNet/scans_uncomp/",
        image_transforms=[],
        image_size = 256,
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
            if self.scene_num_views[scene_id]  < 20*(self.T_in+self.T_out+1):
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
                cond_frame_id_start = random.randint(0, num_frames - 20*(self.T_in+self.T_out+1))
                
            else:
                cond_frame_id_start = current_frame_id
            
            cond_frame_ids = list(
                range(cond_frame_id_start, cond_frame_id_start + self.T_in*20, 20)
                                )
            
            target_frame_ids = list(
                range(cond_frame_id_start + (self.T_in)*20, cond_frame_id_start + (self.T_in+self.T_out)*20, 20)
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
        data['pose_out'] = Pt
        data['pose_out_inv'] = rearrange(torch.linalg.inv(Pt), 'b c d -> b d c')
        data['pose_in'] = Pc
        data['pose_in_inv'] = rearrange(torch.linalg.inv(Pc), 'b c d -> b d c')

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
        image_size = 256,
        mode="train",
        T_in=1,
        T_out=1,
        image_strides=[8,16,32,64]
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
                cond_frame_id_start = random.randint(0, num_frames - 20*(self.T_in+self.T_out+1))
            else:
                cond_frame_id_start = current_frame_id
            
            cond_frame_ids = list(
                range(cond_frame_id_start, cond_frame_id_start + self.T_in*20, 20)
            )
            
            target_frame_ids = list(
                range(cond_frame_id_start + (self.T_in)*20, cond_frame_id_start + (self.T_in+self.T_out)*20, 20)
            )

            Pc = []
            cond_frames = []
            valid_cond = True
            for cond_frame_id in cond_frame_ids:
                frame, P, K= self.load_frame_data(scene_root, cond_frame_id)
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
            cond_pose = Pc[frame_id].numpy() # world2cam

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
        data['pose_out'] = Pt
        data['pose_out_inv'] = rearrange(torch.linalg.inv(Pt), 'b c d -> b d c')
        data['pose_in'] = Pc
        data['pose_in_inv'] = rearrange(torch.linalg.inv(Pc), 'b c d -> b d c')
        for k, v in target_rays.items():
            data[f"target_{k}"] = rearrange(v, "t h w c -> t c h w")
        return data


def crop_img(img: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray]:
    K = K.copy()
    H, W, _ = img.shape
    if H > W:
        margin_l = (H - W) // 2
        margin_r = H - margin_l - W
        img = img[margin_l:-margin_r, :]
        K[1, 2] -= margin_l
    elif H < W:
        margin_l = (W - H) // 2
        margin_r = W - margin_l - H
        img = img[:, margin_l:-margin_r]
        K[0, 2] -= margin_l

    return img, K


def resize_img(img: np.ndarray, K: np.ndarray, size: int) -> Tuple[np.ndarray]:
    H, W, _ = img.shape
    assert H == W

    img = cv2.resize(img, (size, size))
    K = K.copy()
    K *= size / H
    K[2, 2] = 1.0
    return img, K


def get_ray_direction(R: Tensor, K, h: int, w: int, stride: int = 1) -> Tensor:
    """
    get ray direction in world coordinate
    R rotation matrix camera2world
    K intrinsic matrix camera2uv
    h,w image height and width

    return
        ray direction of shape h//stride,w//stride,3
        camera center

    camera coordinate: Y is down, Z is camera direction, X is right
    #camera coordinate: Z is up, Y is camera direction, X is right
    """
    assert h % stride == 0 and w % stride == 0
    K = K / stride
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    xs = torch.arange(w // stride, device=R.device) + 0.5  # shape W,
    xs = (xs - cx) / fx

    ys = torch.arange(h // stride, device=R.device) + 0.5
    ys = (ys - cy) / fy

    ys, xs = torch.meshgrid(ys, xs, indexing="ij")  # shape H,W
    zs = torch.ones_like(ys)
    ray_direction = torch.stack([xs, ys, zs], dim=-1)  # in camera coordinate
    ray_direction = torch.matmul(
        R.view(1, 1, 3, 3), ray_direction.unsqueeze(-1)
    ).squeeze(
        -1
    )  # in world coordinate
    return ray_direction


def get_plucker_coordinate(
    ray_direction: Tensor, ray_origin: Tensor, normalize_d=True
) -> Tensor:
    """
    ray direction: h,w,3
    ray_origin: 3 or h,w,3

    return coordinate h,w,6
    """
    if normalize_d:
        ray_direction = ray_direction / torch.linalg.norm(
            ray_direction, dim=-1, keepdim=True
        )
    if len(ray_origin.size()) == 1:
        h, w, _ = ray_direction.size()
        ray_origin = repeat(ray_origin, "c -> h w c", h=h, w=w)
    m = torch.linalg.cross(ray_origin, ray_direction)  # moment
    return torch.cat([m, ray_direction], dim=-1)
