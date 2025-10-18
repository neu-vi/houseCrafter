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
import quaternion as qt
from typing import List, Dict, Set, Tuple
import zlib
import itertools

# import pyquaternion as qt
from einops import rearrange, repeat
from torch import Tensor
from typing import Tuple
import random
from torchvision import transforms
from tqdm import tqdm
import json
import networkx as nx
from glob import glob

try:
    from .dataset import resize_img, get_ray_direction, get_plucker_coordinate
    from .cfg_util import get_obj_from_str
except ImportError:
    from dataset import resize_img, get_ray_direction, get_plucker_coordinate
    from cfg_util import get_obj_from_str
import lmdb

# in get_ray_direction
# the camera coordinate is Y is down, Z is camera direction, X is right
# position in matterport coordinate is Z up, Y is camera direction, X is right
# base matrix to transform camera coordinate to matterport coordinate
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
        dataset_class: str,
        val_dir=None,
        image_size=256,
        available_pano_file: str = None,
        max_rot_step=4,
        min_rot_step=2,
        dist_threshold=3.0,  # in meters
        num_workers=4,
    ):

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.graph_dir = graph_dir
        self.num_workers = num_workers
        self.image_strides = image_strides
        self.image_strides = image_strides
        self.image_size = image_size
        self.max_rot_step = max_rot_step
        self.min_rot_step = min_rot_step
        self.available_pano_file = available_pano_file
        self.dist_threshold = dist_threshold
        self.dataset_cls = dataset_class
        assert dataset_class in ["mp_dataset.RCNMatterport", "mp_dataset.RCNMatterportExt"]
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
        )
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
        )


class RCNMatterport(Dataset):
    ROTATION_ANGLE = np.radians(15)

    def __init__(
        self,
        image_strides: List[int],
        graph_dir,
        root_dir="/work/vig/Datasets/ScanNet/scans_uncomp/",
        image_transforms=[],
        image_size=256,
        mode="train",
        max_rot_step=4,
        min_rot_step=2,
        available_pano_file=None,
        dist_threshold=3.0,  # in meters
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
                if available_pano and (
                    node1 not in nodes_set or node2 not in nodes_set
                ):
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
        self.items_meta = single_node_meta + two_nodes_meta

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

    def __len__(self):
        return len(self.items_meta)

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
                out["cond"] = [(pano_id, 0), (pano_id, 4), (pano_id, 8)]
                out["target"] = [(pano_id, 12), (pano_id, 16), (pano_id, 20)]
            else:
                offset = np.random.randint(0, 4)
                base = np.random.permutation(6)
                indices = base * 4 + offset
                meta = [(pano_id, i) for i in indices]
                out["cond"] = meta[:3]
                out["target"] = meta[3:]

        elif len(item_meta) == 2:
            if self.mode != "train":
                pano_id1, pano_id2 = item_meta
                out["cond"] = [(pano_id1, 0), (pano_id1, 4), (pano_id1, 8)]
                out["target"] = [(pano_id2, 0), (pano_id2, 4), (pano_id2, 8)]
            else:
                pano_id1, pano_id2 = np.random.permutation(item_meta)
                start = np.random.randint(24)
                steps = np.random.randint(
                    self.min_rot_step, self.max_rot_step + 1, size=2
                )
                indices = [start, start + steps[0], start + steps[0] + steps[1]]
                indices = [i % 24 for i in indices]
                out["cond"] = [(pano_id1, i) for i in indices]
                out["target"] = [(pano_id2, i) for i in indices]
        return out

    def load_frame_data(self, scene_id, pano_id: str, frame_id: int):
        """
        load img and preprocess
        return img, pose (world2cam)
        """
        # load image
        img_path = f"{self.data_dir}/{scene_id}/{pano_id}.{frame_id:0>2}.jpeg"
        img = plt.imread(img_path)
        img = self.preprocess_img(img)

        # get pose matrix
        position = self.graphs[scene_id][pano_id]["position"]
        pose = np.array(
            [position[0], position[1], position[2], frame_id * self.ROTATION_ANGLE, 0]
        )
        P = get(pose)
        P = torch.linalg.inv(P)
        return img, P

    def preprocess_img(self, img):
        # img, K = crop_img(img, K)
        if img.shape[0] != self.image_size:
            img = resize_img(img, self.image_size)
        img = Image.fromarray(img)

        img = img.convert("RGB")
        return self.tform(img)

    def __getitem__(self, index):
        scene_id, pano_ids = self.items_meta[index]
        frames_meta = self._sample_frames_meta(pano_ids)
        return self.get_item_from_meta(scene_id, frames_meta)

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
        cond_pose = Pc[frame_id].numpy()  # world2cam

        Pt = []
        tg_frames = []
        target_rays = {f"ray_{stride}": [] for stride in self.image_strides}
        for tg_meta in frames_meta["target"]:
            pano_id, frame_id = tg_meta
            img, pose = self.load_frame_data(pano_id, frame_id)
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

        return data


class RCNMatterportExt(RCNMatterport):
    DB_IMAGE_SIZE = 256
    HEADINGS = np.radians(np.arange(12) * 30)
    ELEVATIONS = np.radians(np.array([-30, 0, 30.0]))
    VIEWS_PER_PANO = 36

    def __init__(
        self,
        image_strides: List[int],
        graph_dir,
        root_dir="/work/vig/Datasets/ScanNet/scans_uncomp/",
        image_transforms=[],
        image_size=256,
        mode="train",
        max_rot_step=4,
        min_rot_step=2,
        available_pano_file=None,
        dist_threshold=3.0,  # in meters
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
                if available_pano and (
                    node1 not in nodes_set or node2 not in nodes_set
                ):
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
                neighbors_pair = sorted(list(itertools.combinations(neighbors, 2)))
                three_nodes_meta.extend(
                    [(scene, [n1, mid_node, n2]) for n1, n2 in neighbors_pair]
                )
                #

        print(f"got {len(single_node_meta)} panos")
        print(f"got {len(two_nodes_meta)} pano pair")
        print(f"got {len(three_nodes_meta)} pano triple")

        self.items_meta = single_node_meta + two_nodes_meta + three_nodes_meta

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

    def load_frames_data(self, scene_id, pano_ids: List[str], frame_ids: List[int]):
        # load rgb images
        db = lmdb.open(
            os.path.join(self.data_dir, scene_id),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        pano_imgs = {}
        with db.begin() as txn:
            for pano_id in set(pano_ids):
                key = pano_id.encode("ascii")
                data = txn.get(key)
                # load rgba image
                data = np.frombuffer(zlib.decompress(data), dtype=np.uint8).reshape(
                    self.VIEWS_PER_PANO, self.DB_IMAGE_SIZE, self.DB_IMAGE_SIZE, 4
                )
                # dont use alpha channel
                pano_imgs[pano_id] = data[..., :3]

        db.close()
        out_images = []
        for pano_id, frame_id in zip(pano_ids, frame_ids):
            img = pano_imgs[pano_id][frame_id]
            img = self.preprocess_img(img)
            out_images.append(img)

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

        # compute rays
        ## select a condition pose
        frame_id = np.random.randint(len(Pc)) if self.mode == "train" else 0
        cond_pose = Pc[frame_id]  # world2cam

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

        return data


def get_c2w(pose):
    """
    pose: center heading elevation
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
