import json
import os
import zlib
from glob import glob
from itertools import chain, combinations, zip_longest
from typing import Dict, List, Set, Tuple

import lmdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import quaternion as qt
import torch
from cfg_util import get_obj_from_str

# import pyquaternion as qt
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .data_utils import get_plucker_coordinate, get_ray_direction, resize_img

# in get_ray_direction
# the camera coordinate is Y is down, Z is camera direction, X is right
# camera coordinate in matterport is Z up, Y is camera direction, X is right
# in world coordinate of matterport Z is up
# base matrix to transform camera coordinate of get_ray_direction to matterport cam
BASE_MATRIX = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)
FOV = 90


class RCNMatterportDataModule:
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

        assert dataset_cls.split(".")[-1] in [
            "RCNMatterport",
            "RCNMatterportExt",
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
        )
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
        )
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
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
        )
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
        )


class RCNMatterport(Dataset):
    ROTATION_ANGLE = np.radians(15)
    NUM_VIEWS = 24

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
    ):
        """
        load graph,
        store edges and pano list

        """
        self.data_dir = root_dir
        self.target_dir = target_dir if target_dir is not None else root_dir
        self.mode = mode
        self.tform = image_transforms
        self.image_strides = image_strides
        self.image_size = image_size
        self.max_rot_step = max_rot_step
        self.min_rot_step = min_rot_step
        self.cond_image_strides = cond_image_strides
        self.cond_image_size = cond_image_size

        scenes = sorted(os.listdir(root_dir))
        print(f"{mode} Found {len(scenes)} scenes")
        self.graphs = load_nav_graphs(graph_dir, scenes)

        if available_pano_file is not None:
            available_pano = json.load(open(available_pano_file, "r"))
            available_pano = {k: set(v) for k, v in available_pano.items()}
        else:
            available_pano = None

        single_node_meta = []
        two_nodes_meta = []
        for scene in scenes:
            graph = self.graphs[scene]
            nodes_set = set(graph.nodes)
            if available_pano:
                nodes_set = nodes_set.intersection(available_pano[scene])

            # make single node meta
            nodes = sorted(list(nodes_set))
            single_node_meta.extend([(scene, [node]) for node in nodes])

            # make 2 node meta
            edges = sorted(list(graph.edges))
            for node1, node2 in edges:
                if node1 not in nodes_set or node2 not in nodes_set:
                    continue
                loc1 = graph.nodes[node1]["position"]
                loc2 = graph.nodes[node2]["position"]
                dist = np.linalg.norm(np.array(loc1) - np.array(loc2))
                if dist > dist_threshold:
                    continue
                two_nodes_meta.append((scene, [node1, node2]))
        print(
            f"got {len(single_node_meta)} panos and {len(two_nodes_meta)} pano pair with distance less than {dist_threshold}m"
        )

        # interleave 2 lists
        self.items_meta = list(
            filter(
                lambda i: i is not None,
                chain.from_iterable(zip_longest(single_node_meta, two_nodes_meta)),
            )
        )

        # compute intrinsic matrix
        center = image_size / 2
        focal = image_size / 2 / np.tan(np.radians(FOV / 2))
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

    def __len__(self):
        return len(self.items_meta)

    def __getitem__(self, index):
        scene_id, pano_ids = self.items_meta[index]
        frames_meta = self._sample_frames_meta(pano_ids)
        return self.get_item_from_meta(scene_id, frames_meta)

    def _sample_frames_meta(self, item_meta: List[str]):
        """
        sample condition and target from the item_meta

        single pano sample:
            6 frames with 60 deg step cover the full pano
            randomly partition into 3 condition and 3 target

        2 pano sample:
            randomly choose 3 heading (at most 60 deg apart)
            and use these same heading for both pano

        args:
            item_meta: dict specify pano(s) to sample frame from
        return {
                cond: [(pano_id, view_id)],
                target: [(pano_id, view_id)],
            }
        """
        out = {}
        if len(item_meta) == 1:
            pano_id = item_meta[0]
            if self.mode != "train":
                indices = list(range(0, self.NUM_VIEWS, self.NUM_VIEWS // 6))
                out["cond"] = [(pano_id, x) for x in indices[:3]]
                out["target"] = [(pano_id, x) for x in indices[3:]]
            else:
                offset = np.random.randint(0, self.NUM_VIEWS // 6)
                base = np.random.permutation(6)
                indices = base * self.NUM_VIEWS // 6 + offset
                meta = [(pano_id, i) for i in indices]
                out["cond"] = meta[:3]
                out["target"] = meta[3:]

        elif len(item_meta) == 2:
            if self.mode != "train":
                pano_id1, pano_id2 = item_meta
                indices = list(range(0, self.NUM_VIEWS, self.NUM_VIEWS // 6))

                out["cond"] = [(pano_id1, x) for x in indices[:3]]
                out["target"] = [(pano_id2, x) for x in indices[:3]]
            else:
                pano_id1, pano_id2 = np.random.permutation(item_meta)
                start = np.random.randint(self.NUM_VIEWS)
                steps = np.random.randint(
                    self.min_rot_step, self.max_rot_step + 1, size=2
                )
                indices = [start, start + steps[0], start + steps[0] + steps[1]]
                indices = [i % self.NUM_VIEWS for i in indices]
                out["cond"] = [(pano_id1, i) for i in indices]
                out["target"] = [(pano_id2, i) for i in indices]
        return out

    def get_item_from_meta(self, scene_id, frames_meta):
        """
        frames_meta: {
                cond: [(pano_id, view_id)],
                target: [(pano_id, view_id)],
            }
        """
        Pc = []
        cond_frames = []
        for cond_meta in frames_meta["cond"]:
            pano_id, frame_id = cond_meta
            img, pose = self.load_frame_data(scene_id, pano_id, frame_id)
            cond_frames.append(img)
            Pc.append(pose)

        # select a condition pose
        frame_id = np.random.randint(len(Pc)) if self.mode == "train" else 0
        cond_pose = Pc[frame_id]  # world2cam

        # compute ray for cond images
        cond_rays = {f"ray_{stride}": [] for stride in self.cond_image_strides}
        H = W = self.cond_image_size
        for pose, img in zip(Pc, cond_frames):
            ## current_cam 2 cond_cam
            P_rel = cond_pose @ torch.linalg.inv(pose)
            for stride in self.cond_image_strides:
                ray = get_ray_direction(P_rel[:3, :3], self.COND_K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                cond_rays[f"ray_{stride}"].append(flucker_coords)

        Pt = []
        tg_frames = []
        target_rays = {f"ray_{stride}": [] for stride in self.image_strides}
        for tg_meta in frames_meta["target"]:
            pano_id, frame_id = tg_meta
            img, pose = self.load_frame_data(
                scene_id, pano_id, frame_id, is_target=True
            )
            tg_frames.append(img)
            Pt.append(pose)

            # compute ray
            ## target_cam 2 cond_cam
            P_rel = cond_pose @ torch.linalg.inv(pose)
            C, H, W = img.size()
            for stride in self.image_strides:
                ray = get_ray_direction(P_rel[:3, :3], self.K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                target_rays[f"ray_{stride}"].append(flucker_coords)

        cond_frames = torch.stack(cond_frames, dim=0)
        tg_frames = torch.stack(tg_frames, dim=0)
        Pc = torch.stack(Pc, dim=0)
        Pt = torch.stack(Pt, dim=0)

        data = {}
        data["image_input"] = cond_frames
        data["image_target"] = tg_frames
        data["pose_out"] = Pt
        data["pose_out_inv"] = rearrange(torch.linalg.inv(Pt), "b c d -> b d c")
        data["pose_in"] = Pc
        data["pose_in_inv"] = rearrange(torch.linalg.inv(Pc), "b c d -> b d c")

        for k, v in target_rays.items():
            data[f"target_{k}"] = rearrange(v, "t h w c -> t c h w")

        for k, v in cond_rays.items():
            data[f"cond_{k}"] = rearrange(v, "t h w c -> t c h w")

        return data

    def load_frame_data(self, scene_id, pano_id: str, frame_id: int, is_target=False):
        """
        load img and preprocess
        return img, pose (world2cam)
        """
        # load image
        data_dir = self.target_dir if is_target else self.data_dir
        img_path = f"{data_dir}/{scene_id}/{pano_id}.{frame_id:0>2}.jpeg"
        img = plt.imread(img_path)
        img = self.preprocess_img(img)

        # get pose matrix
        position = self.graphs[scene_id].nodes[pano_id]["position"]
        pose = np.array(
            [position[0], position[1], position[2], frame_id * self.ROTATION_ANGLE, 0]
        )
        P = get_c2w(pose)
        P = torch.linalg.inv(P)
        return img, P

    def preprocess_img(self, img):
        # img, K = crop_img(img, K)
        if img.shape[0] != self.image_size:
            img = resize_img(img, self.image_size)
        img = Image.fromarray(img)

        img = img.convert("RGB")
        return self.tform(img)


class RCNMatterportExt(RCNMatterport):
    DB_IMAGE_SIZE = 256
    HEADINGS = np.radians(np.arange(12) * 30)
    ELEVATIONS = np.radians(np.array([-30, 0, 30.0]))
    VIEWS_PER_PANO = 36

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
    ):
        """
        load graph,
        store edges and pano list

        """
        self.data_dir = root_dir
        self.mode = mode
        self.tform = image_transforms
        self.image_strides = image_strides
        self.image_size = image_size
        self.max_rot_step = max_rot_step
        self.min_rot_step = min_rot_step
        self.cond_image_strides = cond_image_strides
        self.cond_image_size = cond_image_size

        scenes = sorted(os.listdir(root_dir))
        print(f"{mode} Found {len(scenes)} scenes")
        self.graphs = load_extended_nav_graphs(graph_dir, scenes)

        if available_pano_file is not None:
            available_pano = json.load(open(available_pano_file, "r"))
            available_pano = {k: set(v) for k, v in available_pano.items()}
        else:
            available_pano = None

        single_node_meta = []
        two_nodes_meta = []
        three_nodes_meta = []
        for scene in scenes:
            graph = self.graphs[scene]
            nodes_set = set(graph.nodes)
            if available_pano:
                nodes_set = nodes_set.intersection(available_pano[scene])

            # make single node meta
            nodes = sorted(list(nodes_set))
            single_node_meta.extend([(scene, [node]) for node in nodes])

            # make 2 node meta
            edges = sorted(list(graph.edges))
            for node1, node2 in edges:
                if node1 not in nodes_set or node2 not in nodes_set:
                    continue
                loc1 = graph.nodes[node1]["position"]
                loc2 = graph.nodes[node2]["position"]
                dist = np.linalg.norm(np.array(loc1) - np.array(loc2))
                if dist > dist_threshold:
                    continue
                two_nodes_meta.append((scene, [node1, node2]))

            # make 3 node meta
            for mid_node in nodes:
                neighbors = list(graph.neighbors(mid_node))
                # filter unavailable nodes
                neighbors = [x for x in neighbors if x in nodes_set]
                # filter by distance
                mid_loc = graph.nodes[mid_node]["position"]
                neighbors = [
                    x
                    for x in neighbors
                    if np.linalg.norm(np.array(graph.nodes[x]["position"]) - mid_loc)
                    < dist_threshold
                ]
                neighbors_pair = sorted(list(combinations(neighbors, 2)))
                three_nodes_meta.extend(
                    [(scene, [n1, mid_node, n2]) for n1, n2 in neighbors_pair]
                )

        print(f"got {len(single_node_meta)} panos")
        print(f"got {len(two_nodes_meta)} pano pair")
        print(f"got {len(three_nodes_meta)} pano triple")

        # interleave 2 lists
        self.items_meta = list(
            filter(
                lambda i: i is not None,
                chain.from_iterable(
                    zip_longest(single_node_meta, two_nodes_meta, three_nodes_meta)
                ),
            )
        )

        # compute intrinsic matrix
        center = image_size / 2
        focal = image_size / 2 / np.tan(np.radians(FOV / 2))
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

    def _sample_frames_meta(self, item_meta: List[str]):
        """
        sample condition and target from the item_meta

        single pano sample:
            6 frames with 60 deg step cover the full pano
            randomly partition into 3 condition and 3 target

        2 pano sample:
            randomly choose 3 heading (at most 60 deg apart)
            and use these same heading for both pano

        3 pano sample:
            randomly choose 3 heading (at most 60 deg apart)
            the 3 target frames from a single pano
            the 3 condition frames randomly choose from the other 2 panos
            but have same 3 headings

        args:
            item_meta: dict specify pano(s) to sample frame from
        return {
                cond: [(pano_id, view_id)],
                target: [(pano_id, view_id)],
            }
        """
        out = {}
        if len(item_meta) == 1:
            pano_id = item_meta[0]
            if self.mode != "train":
                elevation_id = 1
                headings_ids = list(
                    range(0, len(self.HEADINGS), len(self.HEADINGS) // 6)
                )
                frame_ids = [
                    x + elevation_id * len(self.HEADINGS) for x in headings_ids
                ]
                out["cond"] = [(pano_id, x) for x in frame_ids[:3]]
                out["target"] = [(pano_id, x) for x in frame_ids[3:]]
            else:
                # sample headings
                h_offset = np.random.randint(0, len(self.HEADINGS) // 6)
                h_base = np.random.permutation(6)
                h_ids = h_base * len(self.HEADINGS) // 6 + h_offset

                # sample elevations
                e_ids = [1]
                for _ in range(5):
                    min_e = max(0, e_ids[-1] - 1)
                    max_e = min(len(self.ELEVATIONS) - 1, e_ids[-1] + 1)
                    e_ids.append(np.random.randint(min_e, max_e + 1))
                e_ids = np.array(e_ids)

                frame_ids = h_ids + e_ids * len(self.HEADINGS)
                meta = [(pano_id, i) for i in frame_ids]
                out["cond"] = meta[:3]
                out["target"] = meta[3:]

        elif len(item_meta) == 2:
            if self.mode != "train":
                pano_id1, pano_id2 = item_meta
                elevation_id = 1
                headings_ids = list(
                    range(0, len(self.HEADINGS), len(self.HEADINGS) // 6)
                )
                frame_ids = [
                    x + elevation_id * len(self.HEADINGS) for x in headings_ids
                ]
                out["cond"] = [(pano_id1, x) for x in frame_ids[:3]]
                out["target"] = [(pano_id2, x) for x in frame_ids[:3]]
            else:
                # shuffle pano_id
                pano_id1, pano_id2 = np.random.permutation(item_meta)

                # fixed elevation
                elevation_id = 1

                # sample headings
                start = np.random.randint(len(self.HEADINGS))
                steps = np.random.randint(
                    self.min_rot_step, self.max_rot_step + 1, size=2
                )
                h_ids = [start, start + steps[0], start + steps[0] + steps[1]]
                h_ids = [i % len(self.HEADINGS) for i in h_ids]

                frame_ids = [x + elevation_id * len(self.HEADINGS) for x in h_ids]
                out["cond"] = [(pano_id1, i) for i in frame_ids]
                out["target"] = [(pano_id2, i) for i in frame_ids]

        elif len(item_meta) == 3:
            if self.mode != "train":
                pano_id1, pano_id2, pano_id3 = item_meta
                elevation_id = 1
                headings_ids = list(
                    range(0, len(self.HEADINGS), len(self.HEADINGS) // 6)
                )
                frame_ids = [
                    x + elevation_id * len(self.HEADINGS) for x in headings_ids
                ]
                cond_pano_ids = [pano_id1, pano_id2, pano_id2]
                out["cond"] = [
                    (pano_id, x) for pano_id, x in zip(cond_pano_ids, frame_ids[:3])
                ]
                out["target"] = [(pano_id3, x) for x in frame_ids[:3]]
            else:
                # shuffle pano_id
                pano_id1, pano_id2, pano_id3 = np.random.permutation(item_meta)

                # fixed elevation
                elevation_id = 1

                # sample headings
                start = np.random.randint(len(self.HEADINGS))
                steps = np.random.randint(
                    self.min_rot_step, self.max_rot_step + 1, size=2
                )
                h_ids = [start, start + steps[0], start + steps[0] + steps[1]]
                h_ids = [i % len(self.HEADINGS) for i in h_ids]

                frame_ids = [x + elevation_id * len(self.HEADINGS) for x in h_ids]
                cond_pano_ids = [pano_id1, pano_id2, pano_id2]
                out["cond"] = [
                    (pano_id, x) for pano_id, x in zip(cond_pano_ids, frame_ids)
                ]
                out["target"] = [(pano_id3, i) for i in frame_ids]
        return out

    # def load_frames_data(self, scene_id, pano_ids: List[str], frame_ids: List[int]):
    #     # load rgb images
    #     db = lmdb.open(
    #         os.path.join(self.data_dir, scene_id),
    #         readonly=True,
    #         lock=False,
    #         readahead=False,
    #         meminit=False,
    #     )
    #     pano_imgs = {}
    #     with db.begin() as txn:
    #         for pano_id in set(pano_ids):
    #             key = pano_id.encode("ascii")
    #             data = txn.get(key)
    #             # load rgba image
    #             data = np.frombuffer(zlib.decompress(data), dtype=np.uint8).reshape(
    #                 self.VIEWS_PER_PANO, self.DB_IMAGE_SIZE, self.DB_IMAGE_SIZE, 4
    #             )
    #             # dont use alpha channel
    #             pano_imgs[pano_id] = data[..., :3]

    #     db.close()
    #     out_images = []
    #     for pano_id, frame_id in zip(pano_ids, frame_ids):
    #         img = pano_imgs[pano_id][frame_id]
    #         img = self.preprocess_img(img)
    #         out_images.append(img)

    #     # make poses matrices
    #     poses = []
    #     for pano_id, frame_id in zip(pano_ids, frame_ids):
    #         position = self.graphs[scene_id].nodes[pano_id]["position"]
    #         heading = self.HEADINGS[frame_id % len(self.HEADINGS)]
    #         elevation = self.ELEVATIONS[frame_id // len(self.HEADINGS)]
    #         pose = np.array([position[0], position[1], position[2], heading, elevation])
    #         P = get_c2w(pose)
    #         P = torch.linalg.inv(P)
    #         poses.append(P)
    #     return out_images, poses

    def load_frames_data(self, scene_id, pano_ids: List[str], frame_ids: List[int]):
        # load rgb images
        db = lmdb.open(
            os.path.join(self.data_dir, scene_id),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        out_images = []
        with db.begin() as txn:
            for pano_id, frame_id in zip(pano_ids, frame_ids):
                key = f"{pano_id}.{frame_id:0>2}".encode("ascii")
                data = txn.get(key)
                # load rgba image
                data = np.frombuffer(zlib.decompress(data), dtype=np.uint8).reshape(
                    self.DB_IMAGE_SIZE, self.DB_IMAGE_SIZE, 4
                )
                # dont use alpha channel
                data = data[..., :3]
                img = self.preprocess_img(data)
                out_images.append(img)
        db.close()

        # make poses matrices
        poses = []
        for pano_id, frame_id in zip(pano_ids, frame_ids):
            position = self.graphs[scene_id].nodes[pano_id]["position"]
            heading = self.HEADINGS[frame_id % len(self.HEADINGS)]
            elevation = self.ELEVATIONS[frame_id // len(self.HEADINGS)]
            pose = np.array([position[0], position[1], position[2], heading, elevation])
            P = get_c2w(pose)
            P = torch.linalg.inv(P)
            poses.append(P)
        return out_images, poses

    def get_item_from_meta(self, scene_id, frames_meta):
        pano_ids, frame_ids = [], []
        for key in ["cond", "target"]:
            for meta in frames_meta[key]:
                pano_id, frame_id = meta
                pano_ids.append(pano_id)
                frame_ids.append(frame_id)
        images, poses = self.load_frames_data(scene_id, pano_ids, frame_ids)

        Pc = poses[: len(frames_meta["cond"])]
        cond_frames = images[: len(frames_meta["cond"])]

        Pt = poses[len(frames_meta["cond"]) :]
        tg_frames = images[len(frames_meta["cond"]) :]

        ## select a condition pose
        frame_id = np.random.randint(len(Pc)) if self.mode == "train" else 0
        cond_pose = Pc[frame_id]  # world2cam

        # compute ray for cond images
        cond_rays = {f"ray_{stride}": [] for stride in self.cond_image_strides}
        H = W = self.cond_image_size
        for pose in Pc:
            P_rel = cond_pose @ torch.linalg.inv(pose)
            for stride in self.cond_image_strides:
                ray = get_ray_direction(P_rel[:3, :3], self.COND_K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                cond_rays[f"ray_{stride}"].append(flucker_coords)

        # compute ray for target images
        target_rays = {f"ray_{stride}": [] for stride in self.image_strides}
        C, H, W = tg_frames[0].size()
        assert C == 3
        for pose in Pt:
            P_rel = cond_pose @ torch.linalg.inv(pose)
            for stride in self.image_strides:
                ray = get_ray_direction(P_rel[:3, :3], self.K, H, W, stride)
                flucker_coords = get_plucker_coordinate(ray, P_rel[:3, 3])
                target_rays[f"ray_{stride}"].append(flucker_coords)

        cond_frames = torch.stack(cond_frames, dim=0)
        tg_frames = torch.stack(tg_frames, dim=0)
        Pc = torch.stack(Pc, dim=0)
        Pt = torch.stack(Pt, dim=0)

        data = {}
        data["image_input"] = cond_frames
        data["image_target"] = tg_frames
        data["pose_out"] = Pt
        data["pose_out_inv"] = rearrange(torch.linalg.inv(Pt), "b c d -> b d c")
        data["pose_in"] = Pc
        data["pose_in_inv"] = rearrange(torch.linalg.inv(Pc), "b c d -> b d c")

        for k, v in target_rays.items():
            data[f"target_{k}"] = rearrange(v, "t h w c -> t c h w")

        for k, v in cond_rays.items():
            data[f"cond_{k}"] = rearrange(v, "t h w c -> t c h w")
        return data


def get_c2w(pose):
    """
    pose: center heading elevation

    matterport cam coordinate is Z up, Y is camera direction, X is right
    cam coordinate of get_ray_direction is Y down, Z is camera direction, X is right

    the heading and elevation is for matterport cam coordinate
    to make it compatible with get_ray_direction this function use the base matrix
    so that the output pose matrix is in get_ray_direction cam coordinate
    """
    elevation_r = qt.from_rotation_vector(pose[-1] * np.array([1.0, 0.0, 0.0]))

    # positive direction of heading is clockwise
    # zero heading point to Y axis
    heading_r = qt.from_rotation_vector(-pose[-2] * np.array([0.0, 0.0, 1.0]))
    r = qt.as_rotation_matrix(heading_r * elevation_r)

    transform = np.eye(4)
    transform[:3, :3] = r @ BASE_MATRIX
    transform[:3, 3] = pose[:3]
    transform = torch.tensor(transform, dtype=torch.float32)

    return transform


def load_extended_nav_graphs(
    connectivity_dir: str, scans: List[str]
) -> Dict[str, nx.Graph]:
    """Load connectivity graph for each scan"""

    def distance(pose1, pose2):
        """Euclidean distance between two graph poses"""
        return np.linalg.norm(pose1 - pose2)

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, f"{scan}.json")) as f:
            data = json.load(f)
            G = nx.Graph()
            positions = {}
            levels = {}
            # node infor
            for node, node_data in data["nodes"].items():
                positions[node] = np.array(node_data["position"])
                levels[node] = node_data["level"]
            for edge in data["edges"]:
                G.add_edge(
                    edge[0],
                    edge[1],
                    weight=distance(positions[edge[0]], positions[edge[1]]),
                )
            nx.set_node_attributes(G, values=positions, name="position")
            nx.set_node_attributes(G, values=levels, name="level")
            graphs[scan] = G
    return graphs


def load_nav_graphs(connectivity_dir: str, scans: List[str]) -> Dict[str, nx.Graph]:
    """Load connectivity graph for each scan"""

    def distance(pose1, pose2):
        """Euclidean distance between two graph poses"""
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, "%s_connectivity.json" % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            positions[item["image_id"]] = np.array(
                                [item["pose"][3], item["pose"][7], item["pose"][11]]
                            )
                            assert data[j]["unobstructed"][
                                i
                            ], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=distance(item, data[j]),
                            )
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G
    return graphs
