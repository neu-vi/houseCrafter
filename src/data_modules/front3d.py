import json
import os
import pickle
import zlib
from itertools import chain, zip_longest
from typing import Dict, List, Set, Tuple

import cv2
import lmdb
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cfg_util import get_obj_from_str, instantiate_from_config
from einops import rearrange
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

from .data_utils import (
    collate_fn,
    get_connected_subgraphs,
    get_plucker_coordinate,
    get_ray_direction,
    get_world_pcd,
    recursive_local_expand,
    resize_img,
)

# in get_ray_direction
# the camera coordinate is Y is down, Z is camera direction, X is right
# camera coordinate of 3dfront dataset is Y up, X right, Z backward
# in world coordinate of 3dfront Y up
# the base matrix transform camera coordinate of get_ray_direction to 3dfront cam
FRONT3D_BASE_MATRIX = np.array(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=np.float32,
)

FOV = 90


class LayoutRCNFront3DDataModule:
    def __init__(
        self,
        image_strides: List[int],
        batch_size,
        graph_dir,
        train_dir,
        train_scene_ids,
        val_scene_ids,
        dataset_cls: str,
        pose_dir=None,
        location_dir=None,
        val_dir=None,
        image_size=256,
        num_workers=4,
        cond_image_strides=[],
        cond_image_size=224,
        n_cond_frames=3,
        n_target_frames=3,
        depth_shape_db=(512, 512),
        load_target_depth=False,
        load_cond_depth=False,
        return_depth_input=False,
        depth_scale=1000.0,
        max_rot_step=2,
        min_rot_step=1,
        depth_tform_cfg: Dict = None,
        # layout args
        layout_shape_db=(512, 512),
        layout_dir: str = "",
        max_num_points=20,
        intersection_pos="height",
        train_depth_dir=None,
        val_depth_dir=None,
        enable_remapping_cls=False,
        layout_label_ids=None,
        n_scenes=None,
        data_root="/projects/vig/Datasets/3D-Front/render_data",
        **kwargs,
    ):

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pose_dir = pose_dir
        self.location_dir = location_dir
        self.batch_size = batch_size
        self.graph_dir = graph_dir
        self.num_workers = num_workers
        self.image_strides = image_strides
        self.image_size = image_size
        self.dataset_cls = dataset_cls
        self.cond_image_strides = cond_image_strides
        self.cond_image_size = cond_image_size
        self.n_cond_frames = n_cond_frames
        self.n_target_frames = n_target_frames
        self.depth_shape_db = depth_shape_db
        self.load_target_depth = load_target_depth
        self.load_cond_depth = load_cond_depth
        self.return_depth_input = return_depth_input
        self.depth_scale = depth_scale
        self.max_rot_step = max_rot_step
        self.min_rot_step = min_rot_step
        self.enable_remapping_cls = enable_remapping_cls
        self.layout_label_ids = layout_label_ids
        self.n_scenes = n_scenes
        self.data_root = data_root

        self.train_scene_ids = json.load(open(train_scene_ids, "r"))
        self.val_scene_ids = json.load(open(val_scene_ids, "r"))
        self.train_depth_dir = train_depth_dir
        self.val_depth_dir = val_depth_dir

        # layout args
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos

        assert dataset_cls.split(".")[-1] in [
            "Front3DPose",
            "Front3DLocation",
            "Front3DDepthOnly",
        ]
        self.tform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        if load_target_depth or dataset_cls.split(".")[-1] == "Front3DDepthOnly":
            self.o_depth_tform = transforms.Compose(
                [
                    lambda x: torch.tensor(x, dtype=torch.float32),
                    instantiate_from_config(depth_tform_cfg),
                ]
            )
        else:
            self.o_depth_tform = None

    def train_dataloader(self):
        dataset = get_obj_from_str(self.dataset_cls)(
            image_strides=self.image_strides,
            graph_dir=self.graph_dir,
            root_dir=self.train_dir,
            scene_ids=self.train_scene_ids,
            pose_dir=self.pose_dir,
            location_dir=self.location_dir,
            image_transforms=self.tform,
            image_size=self.image_size,
            mode="train",
            cond_image_strides=self.cond_image_strides,
            cond_image_size=self.cond_image_size,
            n_cond_frames=self.n_cond_frames,
            n_target_frames=self.n_target_frames,
            depth_shape_db=self.depth_shape_db,
            load_target_depth=self.load_target_depth,
            load_cond_depth=self.load_cond_depth,
            depth_scale=self.depth_scale,
            o_depth_tform=self.o_depth_tform,
            max_rot_step=self.max_rot_step,
            min_rot_step=self.min_rot_step,
            # layout args
            layout_shape_db=self.layout_shape_db,
            layout_dir=self.layout_dir,
            max_num_points=self.max_num_points,
            intersection_pos=self.intersection_pos,
            depth_data_dir=self.train_depth_dir,
            enable_remapping_cls=self.enable_remapping_cls,
            layout_label_ids=self.layout_label_ids,
            n_scenes=self.n_scenes,
            data_root=self.data_root,
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
            graph_dir=self.graph_dir,
            root_dir=self.train_dir,
            scene_ids=self.train_scene_ids,
            pose_dir=self.pose_dir,
            location_dir=self.location_dir,
            image_transforms=self.tform,
            image_size=self.image_size,
            mode="val",
            cond_image_strides=self.cond_image_strides,
            cond_image_size=self.cond_image_size,
            n_cond_frames=self.n_cond_frames,
            n_target_frames=self.n_target_frames,
            depth_shape_db=self.depth_shape_db,
            load_target_depth=self.load_target_depth,
            load_cond_depth=self.load_cond_depth,
            return_depth_input=self.return_depth_input,
            depth_scale=self.depth_scale,
            o_depth_tform=self.o_depth_tform,
            max_rot_step=self.max_rot_step,
            min_rot_step=self.min_rot_step,
            # layout args
            layout_shape_db=self.layout_shape_db,
            layout_dir=self.layout_dir,
            max_num_points=self.max_num_points,
            intersection_pos=self.intersection_pos,
            depth_data_dir=self.train_depth_dir,
            enable_remapping_cls=self.enable_remapping_cls,
            layout_label_ids=self.layout_label_ids,
            n_scenes=self.n_scenes,
            data_root=self.data_root,
        )

        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            collate_fn=collate_fn,
        )

    def val_dataloader(self, step=1, n_workers=1):
        dataset = get_obj_from_str(self.dataset_cls)(
            image_strides=self.image_strides,
            graph_dir=self.graph_dir,
            root_dir=self.val_dir,
            scene_ids=self.val_scene_ids,
            pose_dir=self.pose_dir,
            location_dir=self.location_dir,
            image_transforms=self.tform,
            image_size=self.image_size,
            mode="val",
            cond_image_strides=self.cond_image_strides,
            cond_image_size=self.cond_image_size,
            n_cond_frames=self.n_cond_frames,
            n_target_frames=self.n_target_frames,
            depth_shape_db=self.depth_shape_db,
            load_target_depth=self.load_target_depth,
            load_cond_depth=self.load_cond_depth,
            return_depth_input=self.return_depth_input,
            depth_scale=self.depth_scale,
            o_depth_tform=self.o_depth_tform,
            max_rot_step=self.max_rot_step,
            min_rot_step=self.min_rot_step,
            # layout args
            layout_shape_db=self.layout_shape_db,
            layout_dir=self.layout_dir,
            max_num_points=self.max_num_points,
            intersection_pos=self.intersection_pos,
            step=step,
            depth_data_dir=self.val_depth_dir,
            enable_remapping_cls=self.enable_remapping_cls,
            layout_label_ids=self.layout_label_ids,
            n_scenes=self.n_scenes,
            data_root=self.data_root,
        )

        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            collate_fn=collate_fn,
        )


class Front3DPose(Dataset):
    """
    graph and image view are sample at the pose level
    not the location level with pano direction like Front3dLocation
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
        curation_meta = None,
        inference_ppl = False,
        data_root="/projects/vig/Datasets/3D-Front/render_data",
        **kwargs,
    ):

        self.data_dir = root_dir
        self.target_dir = target_dir if target_dir is not None else root_dir
        self.mode = mode
        self.tform = image_transforms
        self.image_strides = image_strides
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.depth_shape_db = depth_shape_db
        self.load_target_depth = load_target_depth
        self.o_depth_tform = o_depth_tform
        self.load_cond_depth = load_cond_depth
        self.cond_image_strides = cond_image_strides
        self.cond_image_size = cond_image_size
        self.n_cond_frames = n_cond_frames
        self.n_target_frames = n_target_frames
        # test code, delete the following line later

        self.return_depth_input = return_depth_input
        # layout var
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos
        self.enable_remapping_cls = enable_remapping_cls
        self.depths_mask_range = depths_mask_range
        self.curation_meta = curation_meta
        self.data_root = data_root

        if layout_label_ids is not None:
            label_id_mapping = pd.read_csv(layout_label_ids)
            self.label_name2id = label_id_mapping.set_index("name").to_dict()["id"]
            self.label_id2name = label_id_mapping.set_index("id").to_dict()["name"]
        # load graph
        # filter invalid pose from pose_dir
        # make subgraph of size 6 from graphs
        # store list of subgraphs and corresponding scene_id
        if isinstance(scene_ids, str):
            scene_ids = json.load(open(scene_ids, "r"))
            

        if isinstance(scene_ids, list):
            scene_ids = sorted(scene_ids)
            exist_scenes = set(os.listdir(self.data_dir))
            scene_ids = [
                scene_id for scene_id in scene_ids if scene_id in exist_scenes
            ]  # [:20]
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]
            self.scene_paths = None
        else:
            print('scene_paths loaded from json')
            self.scene_paths = scene_ids  # scene_id -> {img_path, layout_path}
            scene_ids = sorted(list(scene_ids.keys()))  # [:20]
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]


        if depth_data_dir is not None and depth_data_dir.endswith(".json"):
            self.depth_data_dir = json.load(open(depth_data_dir, "r"))
        else:
            self.depth_data_dir = depth_data_dir

        print(f"{mode} Found {len(scene_ids)} scenes")
        print('graph_dir:', graph_dir)
        if not os.path.exists(graph_dir):
            graph_dir = os.path.join(self.data_root, 'graph_poses_all')
        graphs = {
            # scene_id: nx.read_gpickle(os.path.join(graph_dir, f"{scene_id}.pkl"))
            scene_id: pickle.load(
                open(os.path.join(graph_dir, f"{scene_id}.pkl"), "rb")
            )
            for scene_id in scene_ids
        }
        self.scene_ids = scene_ids
        self.unfiltered_graphs = graphs.copy()

        # filter invalid pose from pose_dir
        if pose_dir is not None:
            for scene_id in scene_ids:
                invalid_poses_path = os.path.join(pose_dir, f"{scene_id}_invalid.json")
                if os.path.exists(invalid_poses_path):
                    with open(invalid_poses_path, "r") as f:
                        invalid_poses = set(json.load(f))
                    nodes = list(set(graphs[scene_id].nodes) - invalid_poses)
                    graphs[scene_id] = graphs[scene_id].subgraph(nodes)  # .copy()

        items_meta = []
        # since the number of subgraphs of size 6 is large (1m subgraphs)
        # and it is slow to get all subgraphs
        # here each item in items_meta contains scene_id and a start node
        # from this start node, we can sample a subgraph of size 6
        total_frames = self.n_cond_frames + self.n_target_frames
        print('sorting connected components larger than size:', total_frames)
        n_scene = 5
        i = 0
        # self.generate_curation()
        self.complete_graphs = {}


        for scene_id in scene_ids:
            # filter out connected component with size less than required
            print(scene_id)
            G = graphs[scene_id]
            graph = nx.Graph(G) 
            # if doing data curation, then remove the nodes with wall only
            # print('curation_meta:', curation_meta)
            if curation_meta is not None:
                if os.path.exists(os.path.join(curation_meta, f"{scene_id}.json")):
                    curations = json.load(open(os.path.join(curation_meta, f"{scene_id}.json"), "r"))
                    # filter out nodes with wall only
                    nodes = list(graph.nodes)
                    for node in nodes:
                        if curations[str(node)] is None:
                            graph.remove_node(node)
                            continue
                        if curations[str(node)] < 0.35:
                            graph.remove_node(node)
                    # print('curated scene_id:', scene_id)
            self.inference_ppl = inference_ppl
            if self.inference_ppl == True:
                # print('inference_ppl:', self.inference_ppl)
                cc = [c for c in nx.connected_components(G) if len(c) >= 1]
                wall_nodes_set = sorted(sum([list(c) for c in cc], []))
                if not len(wall_nodes_set):
                    continue
                self.complete_graphs[scene_id] = G.subgraph(wall_nodes_set)
                # for node in wall_nodes_set:
                #     items_meta.append((scene_id, node))


            cc = [c for c in nx.connected_components(graph) if len(c) >= total_frames]
            nodes = sorted(sum([list(c) for c in cc], []))
            if not len(nodes):
                continue
            graphs[scene_id] = graph.subgraph(nodes)  # .copy()
            for node in nodes:
                items_meta.append((scene_id, node))
            # print(f"scene_id: {scene_id} at {self.data_dir}")
            # if i == n_scene:
            #     break
            # i += 1

        self.graphs = graphs
        self.items_meta = items_meta[::step]
        # repermute the items_meta
        self._set_K(image_size, cond_image_size, image_strides)

        if self.mode == "train":
            random.shuffle(self.items_meta)  # Shuffle completely during training
        else:
            random.Random(42).shuffle(self.items_meta)  # Fixed shuffle for reproducibility in testing


        # only execute this for once, delete when we do training

    def generate_curation(self, threshold=0.2):
        """
        generate curation meta for the dataset
        """
        # iterate over all scenes
        # for each scene, get all of its nodes
        self.wall_info_dir = os.path.join(self.data_root, 'wall_info_all')
        if not os.path.exists(self.wall_info_dir):
            os.makedirs(self.wall_info_dir)
        for i in range(len(self.scene_ids)):
            scene_id = self.scene_ids[i]
            if self.scene_paths is None:
                layout_dir = self.layout_dir
            else:
                layout_dir = self.scene_paths[scene_id]["layout_path"]

            save_path = os.path.join(self.wall_info_dir, f"{scene_id}.json")
            if os.path.exists(save_path):
                continue

            if not os.path.exists(os.path.join(layout_dir, scene_id)):
                layout_dir = layout_dir.replace('/work/vig/hieu/3dfront_data', self.data_root)
                if not os.path.exists(os.path.join(layout_dir, scene_id)):
                    print(f"scene_id: {scene_id} not found at {layout_dir}")
                    continue

            wall_info_meta = {}
            graph = self.unfiltered_graphs[scene_id]
            keys = list(graph.nodes)
            # for all key in keys, get the layout
            # iterate over 100 keys at a time
            # for i in range(0, len(keys), 100):
            #     key_batch = keys[i:i+100]
            #     is_wall_only = self.check_wall_only_batch(scene_id, key_batch, layout_dir)
            #     print('key_batch:', len(key_batch))
            #     print('is_wall_only:', (is_wall_only.shape))
            #     print('keys:', key_batch[:10])
            #     print('is_wall_only:', is_wall_only[:10])
            #     break
            # break
                # for key, wall_only in zip(key_batch, is_wall_only):
                #     wall_info_meta[key] = wall_only
            # with open(save_path, "w") as f:
            #     json.dump(wall_info_meta, f)
            # print(f"scene_id: {scene_id} is curated")
            for key in keys:
                print(f"scene_id: {scene_id}, key: {key}")
                is_wall_only = self.check_wall_only(scene_id, key, layout_dir, threshold=0.35)
                wall_info_meta[key] = is_wall_only
            with open(save_path, "w") as f:
                json.dump(wall_info_meta, f)
            print(f"scene_id: {scene_id} is curated")

    def check_wall_only(
            self, 
            scene_id,
            frame_id,
            layout_dir,
            layout_depths: List[torch.FloatTensor] = None,
            layout_clss: List[torch.LongTensor] = None,
            threshold=0.2,
        ):
        H, W = self.layout_shape_db
        depth_scale = self.depth_scale
        assert H == W  # and H % self.image_size == 0
        # shape_scale = H // self.image_size * min(self.image_strides)
        layout_size = self.image_size // min(self.image_strides)
        layout = {"layout_cls": [], "layout_pos": []}

        if layout_dir is not None:
            assert scene_id is not None
            assert frame_id is not None
            env = lmdb.open(
                os.path.join(layout_dir, scene_id),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            txn = env.begin(write=False)
        # todo: figure out this part later
        else:
            assert layout_depths is not None
            assert layout_clss is not None
            frame_ids = [None] * len(layout_depths)
        
        if layout_dir is not None:
            key = frame_id.encode("ascii")
            data = txn.get(key)
            # print('data:', data)
            if data is None:
                print(f"frame_id: {frame_id} not found")
                return
            data = zlib.decompress(data)
            data = np.frombuffer(data, dtype=np.uint16).reshape(-1, 2, H, W)
            data = torch.tensor(data.astype(np.int32), dtype=torch.float32)
            data = F.interpolate(data, size=layout_size, mode="nearest")
            layout_cls = data[:, 0].long()
            depth = data[:, 1] / depth_scale
        # todo: figure out where this mode is used, it seems like we don't need it for training
        else:
            depth = layout_depths
            layout_cls = layout_clss
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
        if self.enable_remapping_cls:
            layout_cls[0] = self.remapping_wall_cls(layout_cls[0])
        w_cls = layout_cls[:1]
        w_depth = depth[:1]

        f_cls = layout_cls[1:]
        f_depth = depth[1:]
        # todo: if wall takes up too much of the image, we should label it as a wall only frame
        furniture_type, _, _ = f_cls.shape
        if furniture_type == 0:
            # print('wall only frame')
            return 0
        else:
            #check the percentage of none-zero pixel in f_cls:
            # Step 1: Compute the union mask (occupied space)
            occupied_mask = f_cls.sum(dim=0) > 0  # Element-wise OR across channels

            # Step 2: Calculate the occupied area
            occupied_pixels = occupied_mask.sum().item()
            total_pixels = f_cls.shape[1] * f_cls.shape[2]

            # Step 3: Compute percentage of occupied space
            occupied_percentage = occupied_pixels / total_pixels

            return occupied_percentage

    def check_wall_only_batch(
            self, 
            scene_id,
            frame_ids,
            layout_dir,
            layout_depths: torch.FloatTensor = None,
            layout_clss: torch.LongTensor = None,
            threshold=0.2,
        ):
        H, W = self.layout_shape_db
        depth_scale = self.depth_scale
        assert H == W  
        layout_size = self.image_size // min(self.image_strides)

        if layout_dir is not None:
            assert scene_id is not None
            assert frame_ids is not None
            env = lmdb.open(
                os.path.join(layout_dir, scene_id),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            txn = env.begin(write=False)
            batch_data = []
            
            for frame_id in frame_ids:
                key = frame_id.encode("ascii")
                data = txn.get(key)
                if data is None:
                    print(f"frame_id: {frame_id} not found")
                    batch_data.append(None)
                    continue
                data = zlib.decompress(data)
                data = np.frombuffer(data, dtype=np.uint16).reshape(-1, 2, H, W)
                batch_data.append(torch.tensor(data.astype(np.int32), dtype=torch.float32))
            
            batch_data = [d for d in batch_data if d is not None]
            if not batch_data:
                return None  # No valid frames found
            
            data = torch.stack(batch_data)  # Shape: (batch_size, num_channels, 2, H, W)
            data = F.interpolate(data, size=layout_size, mode="nearest")
            layout_cls = data[:, :, 0].long()
            depth = data[:, :, 1] / depth_scale
        else:
            assert layout_depths is not None
            assert layout_clss is not None
            depth = layout_depths
            layout_cls = layout_clss
            if depth.size(2) != layout_size:
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

        if self.enable_remapping_cls:
            layout_cls[:, 0] = self.remapping_wall_cls(layout_cls[:, 0])

        w_cls = layout_cls[:, :1]
        w_depth = depth[:, :1]
        f_cls = layout_cls[:, 1:]
        f_depth = depth[:, 1:]

        furniture_type, batch_size, _, _ = f_cls.shape
        if furniture_type == 0:
            return torch.zeros(batch_size)  # No furniture, return zero occupancy

        # Batch-wise computation of occupied space
        occupied_mask = f_cls.sum(dim=1) > 0  # OR operation across furniture channels
        occupied_pixels = occupied_mask.view(batch_size, -1).sum(dim=1)  # Sum per frame
        total_pixels = f_cls.shape[2] * f_cls.shape[3]  # Total pixels per frame

        # Compute percentage occupancy for each frame
        occupied_percentage = occupied_pixels.float() / total_pixels

        return occupied_percentage


    def log_frame_vis(self, scene_id, frame_ids, logging_dir,is_target=False,node_batch=False):
        """
        log image, pose, layout, depth, ray, pcd, mask
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

        if self.depth_data_dir is not None:
            if isinstance(self.depth_data_dir, str):
                depth_path = os.path.join(self.depth_data_dir, scene_id)
            else:
                depth_path = os.path.join(
                    self.depth_data_dir[scene_id]["depth_path"], scene_id
                )
            depth_db = lmdb.open(
                depth_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        else:
            depth_db = db
        
        # preparing save_path
        if node_batch is True:
            save_path = os.path.join(logging_dir, f'{scene_id}_{frame_ids[0]}')
        else:
            save_path = os.path.join(logging_dir, f'{scene_id}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        rgb_save_path = os.path.join(save_path, "rgb")
        if not os.path.exists(rgb_save_path):
            os.makedirs(rgb_save_path)
        pose_save_path = os.path.join(save_path, "pose")
        if not os.path.exists(pose_save_path):
            os.makedirs(pose_save_path)
        depth_save_path = os.path.join(save_path, "depth")
        if not os.path.exists(depth_save_path):
            os.makedirs(depth_save_path)

        with db.begin(write=False) as txn:
            for frame_id in frame_ids:
                rgb_key = f"{frame_id}_rgb".encode("ascii")
                img = txn.get(rgb_key)
                img = np.frombuffer(img, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                self.save_img(img, os.path.join(rgb_save_path, f"{frame_id}.png"))

                pose_key = f"{frame_id}_pose".encode("ascii")
                c2w = txn.get(pose_key)
                c2w = (
                    np.frombuffer(zlib.decompress(c2w), dtype=np.float32)
                    .reshape(4, 4)
                    .copy()
                )
                c2w[3, 3] = 1.0  # bug when save pose
                w2c = np.linalg.inv(c2w)
                np.save(os.path.join(pose_save_path, f"{frame_id}.npy"), w2c)

        with depth_db.begin(write=False) as txn:
            depths = []
            for frame_id in frame_ids:
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
                # normalize and save
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                depth_img = (depth * 255).astype(np.uint8)
                self.save_img(depth_img, os.path.join(depth_save_path, f"{frame_id}.png"))
    
    def save_img(self, img, save_path):
        # img, K = crop_img(img, K)
        if img.shape[0] != self.image_size:
            img = resize_img(img, self.image_size)
        img = Image.fromarray(img)

        img = img.convert("RGB")
        img.save(save_path)


    def _set_K(self, image_size, cond_image_size, image_strides):
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

        self.LAYOUT_K = self.K / min(image_strides)
        self.LAYOUT_K[2, 2] = 1.0

    def __len__(self):
        return len(self.items_meta)

    def __getitem__(self, index):
        scene_id, start_node = self.items_meta[index]
        frames_meta = self._sample_frames_meta(scene_id, start_node)
        data = self.get_item_from_meta(scene_id, frames_meta)
        if self.scene_paths is None:
            layout_dir = self.layout_dir
        else:
            layout_dir = self.scene_paths[scene_id]["layout_path"]
        data.update(
            self.get_layout(
                scene_id=scene_id,
                frame_ids=frames_meta["target"],
                poses=data["pose_out"],
                layout_dir=layout_dir,
            )
        )
        if self.mode != "train":
            data["start_node"] = start_node
        return data

    def get_item_from_meta(
        self,
        scene_id: str,
        frames_meta: Dict[str, List[str]],
        load_target_GT=True,
        target_pose=None,
        anchor=False,
    ):
        """
        args: frames_meta: {
                cond: [view_id],
                target: [view_id],
            }
            target_poses: list of  w2c 4,4 tensor

        return {
            "image_input": shape (num_cond, 3, H, W),
            "image_target": shape (num_target, 3, H, W),
            "pose_out": shape (num_target, 4, 4) w2c of target frames
            "pose_out_inv": shape (num_target, 4, 4) transpose of inverse of pose_out
            "pose_in": shape (num_cond, 4, 4) w2c of cond frames
            "pose_in_inv": shape (num_cond, 4, 4) transpose of inverse of pose_in
            "depth_input": shape (num_cond, H, W), optional, NOTE subject to change
            "depth_target": shape (num_target, H, W), optional, NOTE subject to change
        }
        """
        # load cond frames
        cond_frames, Pc, cond_depths = self.load_frames_data(
            scene_id, frames_meta["cond"], load_depth=self.load_cond_depth, is_target=anchor
        )

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
            tg_frames, Pt, tg_depths = self.load_frames_data(
                scene_id,
                frames_meta["target"],
                load_depth=self.load_target_depth,
                is_target=True,
            )
        else:
            tg_frames = None
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

        cond_frames = torch.stack(cond_frames, dim=0)
        Pc = torch.stack(Pc, dim=0)
        Pt = torch.stack(Pt, dim=0)

        data = {}
        if load_target_GT:
            tg_frames = torch.stack(tg_frames, dim=0)
            data["image_target"] = tg_frames
        data["image_input"] = cond_frames
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

        if self.load_cond_depth:
            if self.return_depth_input:
                _cond_depths = [
                    resize_img(d, self.image_size, interpolation=cv2.INTER_NEAREST)
                    for d in cond_depths
                ]
                _cond_depths = [
                    torch.tensor(d, dtype=torch.float32) for d in _cond_depths
                ]
                data["depth_input"] = rearrange(_cond_depths, "t h w -> t h w")

            cond_depths = [
                resize_img(d, self.cond_image_size, interpolation=cv2.INTER_NEAREST)
                for d in cond_depths
            ]
            cond_depths = [torch.tensor(d, dtype=torch.float32) for d in cond_depths]
            if self.depths_mask_range is not None:
                masks = [
                    d < self.depths_mask_range for d in cond_depths
                ]
                # masks = [torch.tensor(m, dtype=torch.float32) for m in masks]
                data["depth_mask_cond"] = rearrange(masks, "t h w -> t h w")


            # ! assume only one stride for cond depth this may change later
            assert len(self.cond_image_strides) == 1
            c_stride = self.cond_image_strides[0]
            I = torch.eye(4, device=cond_pose.device)
            # get pcd in cam coordinate
            pcd = [
                get_world_pcd(I, self.COND_K, d[None, ...], c_stride)
                for d in cond_depths
            ]
            # strided_depths = [
            #     d[..., c_stride // 2 :: c_stride, c_stride // 2 :: c_stride]
            #     for d in cond_depths
            # ]
            # pcd = rearrange(pcd, "t h w c -> t c h w")
            # strided_depths = rearrange(strided_depths, "t h w -> t 1 h w")
            data["in_pos3d"] = rearrange(pcd, "t 1 h w c -> t h w c")

        if self.load_target_depth and load_target_GT:
            tg_depths = [self.preprocess_output_depth(d) for d in tg_depths]
            if self.depths_mask_range is not None:
                masks = [self.get_mask(d) for d in tg_depths]
                data["depth_mask_target"] = torch.stack(masks, dim=0) # should also be t,h,w
            data["depth_target"] = torch.stack(tg_depths, dim=0)  # t,h,w
        return data

    def _sample_frames_meta(
        self, scene_id: str, start_node: str
    ) -> Dict[str, List[str]]:
        """
        get a subgraph of size 6 from the graph start from the given start_node

        partition frames(nodes of subgraph) into condition and target frames

        return frames_meta: {
                cond: [view_id],
                target: [view_id],
            }

        """
        total_frames = self.n_cond_frames + self.n_target_frames
        # sample subgraph
        subgraphs = []
        exclude = {start_node}
        G = self.graphs[scene_id]
        deterministic = self.mode != "train"
        # load the curation meta
        if self.curation_meta is not None and os.path.exists(os.path.join(self.curation_meta, f"{scene_id}.json")):
            curation_meta = json.load(open(os.path.join(self.curation_meta, f"{scene_id}.json"), "r"))
        else:
            curation_meta = None
        recursive_local_expand(
            G,
            {start_node},
            set(G.neighbors(start_node)) - exclude,
            exclude,
            subgraphs,
            total_frames,
            deterministic,
            return_one=True,
            curation_list=curation_meta,
        )
        frame_ids = sorted(subgraphs[0])
        if self.mode == "train":
            frame_ids = np.random.permutation(frame_ids).tolist()
        # wall_only_frames = [c for c in frame_ids if curation_meta[c] <=0.35]
        # furniture_frames = [c for c in frame_ids if curation_meta[c] > 0.35]
        # self.log_frame_vis(scene_id, 
        #                    wall_only_frames, 
        #                    '/work/vig/hieu/escher/6DoF/inspection/wall_only', 
        #                    is_target=False,)
        # self.log_frame_vis(scene_id,
        #                    frame_ids,
        #                    '/work/vig/hieu/escher/6DoF/inspection/furniture',
        #                    is_target=False,)

        return {
            "cond": frame_ids[: self.n_cond_frames],
            "target": frame_ids[self.n_cond_frames :],
        }

    def load_frames_data(
        self,
        scene_id,
        frame_ids: List[str],
        load_depth=False,
        is_target=False,
        same_depth_img_db=False,
    ):
        """
        load frames data from lmdb: rgb, pose (w2c), depth (optional)
        """
        # if self.scene_paths is None:
        if self.scene_paths is None:
            if is_target:
                data_dir = self.target_dir
            else:
                data_dir = self.data_dir
        # print('data_dir:', data_dir)
        if not os.path.exists(os.path.join(data_dir, scene_id)):
            data_dir = data_dir.replace("/work/vig/hieu/3dfront_data/", self.data_root + "/")
        db = lmdb.open(
            os.path.join(data_dir, scene_id),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if self.depth_data_dir is not None and not same_depth_img_db:
            if isinstance(self.depth_data_dir, str):
                depth_path = os.path.join(self.depth_data_dir, scene_id)
            else:
                depth_path = os.path.join(
                    self.depth_data_dir[scene_id]["depth_path"], scene_id
                )
            if not os.path.exists(depth_path):
                depth_path = depth_path.replace("/work/vig/hieu/3dfront_data/", self.data_root + "/")
            depth_db = lmdb.open(
                depth_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        else:
            depth_db = db

        rgbs, poses = [], []
        with db.begin(write=False) as txn:
            i = 0
            for frame_id in frame_ids:
                rgb_key = f"{frame_id}_rgb".encode("ascii")
                img = txn.get(rgb_key)
                if img is None:
                    print(f"frame_id: {frame_id} not found at {data_dir}")
                    while True:
                        frame_id = int(frame_id) + 1
                        # convert to 4 digit str
                        frame_id = str(frame_id).zfill(4)
                        rgb_key = f"{frame_id}_rgb".encode("ascii")
                        img = txn.get(rgb_key)
                        if img is not None:
                            frame_ids[i] = frame_id
                            break
                img = np.frombuffer(img, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = self.preprocess_img(img)
                rgbs.append(img)

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

        if load_depth:
            with depth_db.begin(write=False) as txn:
                depths = []
                for frame_id in frame_ids:
                    depth_key = f"{frame_id}_depth".encode("ascii")
                    depth_comp = txn.get(depth_key)
                    if depth_comp is None:
                        while True:
                            frame_id = int(frame_id) + 1
                            # convert to 4 digit str
                            frame_id = str(frame_id).zfill(4)
                            depth_key = f"{frame_id}_depth".encode("ascii")
                            depth_comp = txn.get(depth_key)
                            if depth_comp is not None:
                                frame_ids[i] = frame_id
                                break
                    depth = zlib.decompress(depth_comp)
                    depth = np.frombuffer(depth, dtype=np.uint16)
                    size = int(np.sqrt(len(depth)))
                    depth = depth.reshape(size, size)
                    # convert from mm to m in this case depth_scale = 1000
                    depth = depth.astype(np.float32) / self.depth_scale
                    depths.append(depth)
        else:
            depths = [None] * len(frame_ids)

        db.close()
        if depth_db != db:
            depth_db.close()

        return rgbs, poses, depths

    def get_layout(
        self,
        poses,
        scene_id: str = None,
        frame_ids: List[str] = None,
        layout_dir: str = None,
        layout_depths: List[torch.FloatTensor] = None,
        layout_clss: List[torch.LongTensor] = None,
    ) -> Dict:
        """
        load layout from frame meta
        args:
            poses: w2c
        """
        print('get layout')
        # print('get layout by loading data from recorded pcd')
        H, W = self.layout_shape_db
        depth_scale = self.depth_scale
        assert H == W  # and H % self.image_size == 0
        # shape_scale = H // self.image_size * min(self.image_strides)
        layout_size = self.image_size // min(self.image_strides)

        layout = {"layout_cls": [], "layout_pos": []}
        if not os.path.exists(layout_dir):
            layout_dir = layout_dir.replace("/work/vig/hieu/3dfront_data/", self.data_root + "/")

        if layout_dir is not None:
            assert scene_id is not None
            assert frame_ids is not None
            print('accessing layout at:', layout_dir)
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
                if data is None:
                    # print(f"frame_id: {frame_id} not found at {data_dir}")
                    while True:
                        frame_id = int(frame_id) + 1
                        # convert to 4 digit str
                        frame_id = str(frame_id).zfill(4)
                        rgb_key = f"{frame_id}_rgb".encode("ascii")
                        data = txn.get(rgb_key)
                        if data is not None:
                            break
                        frame_ids[i] = frame_id

                data = zlib.decompress(data)
                data = np.frombuffer(data, dtype=np.uint16).reshape(-1, 2, H, W)
                data = torch.tensor(data.astype(np.int32), dtype=torch.float32)
                data = F.interpolate(data, size=layout_size, mode="nearest")
                layout_cls = data[:, 0].long()
                depth = data[:, 1] / depth_scale
            else:
                depth = layout_depths[i]
                layout_cls = layout_clss[i]
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

            # layout_cls = torch.tensor(
            #     data[
            #         :,
            #         0,
            #         ...,
            #         shape_scale // 2 :: shape_scale,
            #         shape_scale // 2 :: shape_scale,
            #     ].astype(np.int32),
            #     dtype=torch.long,
            # )
            # depth = torch.tensor(
            #     data[
            #         :,
            #         1,
            #         ...,
            #         shape_scale // 2 :: shape_scale,
            #         shape_scale // 2 :: shape_scale,
            #     ].astype(np.float32)
            #     / depth_scale
            # )

            # cut down the number of points
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

    def preprocess_img(self, img):
        # img, K = crop_img(img, K)
        if img.shape[0] != self.image_size:
            img = resize_img(img, self.image_size)
        img = Image.fromarray(img)

        img = img.convert("RGB")
        return self.tform(img)

    def preprocess_output_depth(self, depth):
        # resize
        if depth.shape[0] != self.image_size:
            depth = resize_img(depth, self.image_size, interpolation=cv2.INTER_NEAREST)
        # convert to tensor, normalize
        return self.o_depth_tform(depth)
    
    def get_mask(self, depth):
        if depth.shape[0] != self.image_size:
            depth = resize_img(depth, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = depth < self.depths_mask_range
        return mask

    def remapping_wall_cls(self, wall_cls: torch.Tensor) -> torch.Tensor:
        """
        remapping wall cls id
        wall_cls: shape (H, W) long

        """
        WALL_CLASSES = ["wall", "front", "back", "hole", "window", "door", "pocket"]
        assert self.label_name2id is not None
        assert self.label_id2name is not None
        unique_labels = torch.unique(wall_cls)
        unique_labels = [int(x) for x in unique_labels]
        for label_id in unique_labels:
            name = self.label_id2name[label_id]
            if any([ele in name for ele in WALL_CLASSES]):
                name = "wallinner"
            elif "ceiling" in name:
                name = "ceiling"
            elif "floor" in name:
                name = "floor"
            else:
                name = "void"
            new_id = self.label_name2id[name]
            wall_cls[wall_cls == label_id] = new_id
        return wall_cls


class Front3DLocation(Front3DPose):
    """
    graph and image view are sample at the location level
    there are 12 views in each location
    """

    HEADINGS = np.radians(np.arange(12) * 30)
    NUM_VIEWS = 12
    ELEVATION = 0.0

    def __init__(
        self,
        image_strides: List[int],
        graph_dir,
        root_dir,
        scene_ids: List[str],
        location_dir=None,
        image_transforms=[],
        image_size=256,
        mode="train",
        cond_image_strides=[],
        cond_image_size=224,
        target_dir=None,
        n_cond_frames=3,
        n_target_frames=3,
        depth_shape_db=(512, 512),
        load_target_depth=False,
        load_cond_depth=False,
        depth_scale=1000.0,
        max_rot_step=2,
        min_rot_step=1,
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
        data_root="/projects/vig/Datasets/3D-Front/render_data",
        **kwargs,
    ):

        assert n_cond_frames == 3
        assert n_target_frames == 3
        self.data_dir = root_dir
        self.target_dir = target_dir if target_dir is not None else root_dir
        self.mode = mode
        self.tform = image_transforms
        self.image_strides = image_strides
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.depth_shape_db = depth_shape_db
        self.load_target_depth = load_target_depth
        self.load_cond_depth = load_cond_depth
        self.cond_image_strides = cond_image_strides
        self.cond_image_size = cond_image_size
        self.n_cond_frames = n_cond_frames
        self.n_target_frames = n_target_frames
        self.max_rot_step = max_rot_step
        self.min_rot_step = min_rot_step
        # layout var
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos
        self.enable_remapping_cls = enable_remapping_cls
        self.data_root = data_root
        if layout_label_ids is not None:
            label_id_mapping = pd.read_csv(layout_label_ids)
            self.label_name2id = label_id_mapping.set_index("name").to_dict()["id"]
            self.label_id2name = label_id_mapping.set_index("id").to_dict()["name"]
        # load graph
        # filter invalid location from location_dir
        # make subgraph of size 6 from graphs
        # store list of subgraphs and corresponding scene_id
        if isinstance(scene_ids, str):
            scene_ids = json.load(open(scene_ids, "r"))

        if isinstance(scene_ids, list):
            scene_ids = sorted(scene_ids)
            exist_scenes = set(os.listdir(self.data_dir))
            scene_ids = [
                scene_id for scene_id in scene_ids if scene_id in exist_scenes
            ]  # [:20]
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]
            self.scene_paths = None
        else:
            self.scene_paths = scene_ids  # scene_id -> {img_path, layout_path}
            scene_ids = sorted(list(scene_ids.keys()))  # [:20]
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]

        if depth_data_dir is not None and depth_data_dir.endswith(".json"):
            self.depth_data_dir = json.load(open(depth_data_dir, "r"))
        else:
            self.depth_data_dir = depth_data_dir

        print(f"{mode} Found {len(scene_ids)} scenes")
        graphs = {
            # scene_id: nx.read_gpickle(os.path.join(graph_dir, f"{scene_id}.pkl"))
            scene_id: pickle.load(
                open(os.path.join(graph_dir, f"{scene_id}.pkl"), "rb")
            )
            for scene_id in scene_ids
        }

        # filter invalid location from location_dir
        if location_dir is not None:
            for scene_id in scene_ids:
                invalid_locations_path = os.path.join(
                    location_dir, f"{scene_id}_invalid.json"
                )
                if os.path.exists(invalid_locations_path):
                    with open(invalid_locations_path, "r") as f:
                        invalid_locations = set(json.load(f))
                    nodes = list(set(graphs[scene_id].nodes) - invalid_locations)
                    graphs[scene_id] = graphs[scene_id].subgraph(nodes).copy()

        # each item_meta contains scene_id and 1 to 3 nodes
        single_node_meta = []
        two_nodes_meta = []
        three_nodes_meta = []
        for scene_id in scene_ids:
            graph = graphs[scene_id]
            cc = [c for c in nx.connected_components(graph)]

            # get single node items
            nodes = sorted(list(graph.nodes))
            single_node_meta.extend([(scene_id, [node]) for node in nodes])

            # get two nodes items
            cc2 = [c for c in cc if len(c) >= 2]
            nodes2 = sum([list(c) for c in cc2], [])
            graph2 = graph.subgraph(nodes2)
            subgraph2 = get_connected_subgraphs(graph2, 2)
            two_nodes_meta.extend([(scene_id, sorted(list(x))) for x in subgraph2])

            # get three nodes items
            cc3 = [c for c in cc if len(c) >= 3]
            nodes3 = sum([list(c) for c in cc3], [])
            graph3 = graph.subgraph(nodes3)
            subgraph3 = get_connected_subgraphs(graph3, 3)
            three_nodes_meta.extend([(scene_id, sorted(list(x))) for x in subgraph3])

        print(f"got {len(single_node_meta)} panos")
        print(f"got {len(two_nodes_meta)} pano pair")
        print(f"got {len(three_nodes_meta)} pano triple")

        # self.graphs = graphs
        # interleave 3 lists
        self.items_meta = list(
            filter(
                lambda i: i is not None,
                chain.from_iterable(
                    zip_longest(single_node_meta, two_nodes_meta, three_nodes_meta)
                ),
            )
        )
        self._set_K(image_size, cond_image_size, image_strides)

    def _sample_frames_meta(self, nodes: List[str]):
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

        return frames_meta: {
                cond: [view_id],
                target: [view_id],
            }

        """
        out = {}

        if len(nodes) == 1:
            pano_id = nodes[0]
            if self.mode != "train":
                indices = list(range(0, self.NUM_VIEWS, self.NUM_VIEWS // 6))
                out["cond"] = [f"{pano_id}_{x:0>2}" for x in indices[:3]]
                out["target"] = [f"{pano_id}_{x:0>2}" for x in indices[3:]]
            else:
                offset = np.random.randint(0, self.NUM_VIEWS // 6)
                base = np.random.permutation(6)
                indices = base * self.NUM_VIEWS // 6 + offset
                meta = [f"{pano_id}_{x:0>2}" for x in indices]
                out["cond"] = meta[:3]
                out["target"] = meta[3:]

        elif len(nodes) == 2:
            if self.mode != "train":
                pano_id1, pano_id2 = nodes
                indices = list(range(0, self.NUM_VIEWS, self.NUM_VIEWS // 6))

                out["cond"] = [f"{pano_id1}_{x:0>2}" for x in indices[:3]]
                out["target"] = [f"{pano_id2}_{x:0>2}" for x in indices[:3]]
            else:
                pano_id1, pano_id2 = np.random.permutation(nodes)
                start = np.random.randint(self.NUM_VIEWS)
                steps = np.random.randint(
                    self.min_rot_step, self.max_rot_step + 1, size=2
                )
                indices = [start, start + steps[0], start + steps[0] + steps[1]]
                indices = [i % self.NUM_VIEWS for i in indices]
                out["cond"] = [f"{pano_id1}_{x:0>2}" for x in indices]
                out["target"] = [f"{pano_id2}_{x:0>2}" for x in indices]

        elif len(nodes) == 3:
            if self.mode != "train":
                pano_id1, pano_id2, pano_id3 = nodes
                indices = list(range(0, self.NUM_VIEWS, self.NUM_VIEWS // 6))
                cond_pano_ids = [pano_id1, pano_id2, pano_id2]
                out["cond"] = [
                    f"{pano_id}_{x:0>2}"
                    for pano_id, x in zip(cond_pano_ids, indices[:3])
                ]
                out["target"] = [f"{pano_id3}_{x:0>2}" for x in indices[:3]]
            else:
                # shuffle pano_id
                pano_id1, pano_id2, pano_id3 = np.random.permutation(nodes)

                # sample headings
                start = np.random.randint(len(self.HEADINGS))
                steps = np.random.randint(
                    self.min_rot_step, self.max_rot_step + 1, size=2
                )
                indices = [start, start + steps[0], start + steps[0] + steps[1]]
                indices = [i % self.NUM_VIEWS for i in indices]

                cond_pano_ids = [pano_id1, pano_id2, pano_id2]
                out["cond"] = [
                    f"{pano_id}_{x:0>2}" for pano_id, x in zip(cond_pano_ids, indices)
                ]
                out["target"] = [f"{pano_id3}_{x:0>2}" for x in indices]
        return out

    def __getitem__(self, index):
        scene_id, nodes = self.items_meta[index]
        frames_meta = self._sample_frames_meta(nodes)
        data = self.get_item_from_meta(scene_id, frames_meta)
        if self.scene_paths is None:
            layout_dir = self.layout_dir
        else:
            layout_dir = self.scene_paths[scene_id]["layout_path"]
        data.update(
            self.get_layout(
                scene_id=scene_id,
                frame_ids=frames_meta["target"],
                poses=data["pose_out"],
                layout_dir=layout_dir,
            )
        )
        return data


class Front3DLocationGraph(Front3DPose):
    def __init__(
        self,
        image_strides: List[int],
        graph_dir,
        root_dir,
        scene_ids: List[str],
        image_transforms=[],
        image_size=256,
        mode="train",
        cond_image_strides=[],
        cond_image_size=224,
        target_dir=None,
        n_cond_frames=3,
        n_target_frames=3,
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
        n_headings=5,
        n_elevations=2,
        data_root="/projects/vig/Datasets/3D-Front/render_data",
        **kwargs,
    ):
        assert n_cond_frames == 3
        assert n_target_frames == 3
        self.data_dir = root_dir
        self.target_dir = target_dir if target_dir is not None else root_dir
        self.mode = mode
        self.tform = image_transforms
        self.image_strides = image_strides
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.depth_shape_db = depth_shape_db
        self.load_target_depth = load_target_depth
        self.load_cond_depth = load_cond_depth
        self.o_depth_tform = o_depth_tform
        self.cond_image_strides = cond_image_strides
        self.cond_image_size = cond_image_size
        self.n_cond_frames = n_cond_frames
        self.n_target_frames = n_target_frames
        self.return_depth_input = return_depth_input
        # layout var
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos
        self.enable_remapping_cls = enable_remapping_cls
        self.data_root = data_root
        if layout_label_ids is not None:
            label_id_mapping = pd.read_csv(layout_label_ids)
            self.label_name2id = label_id_mapping.set_index("name").to_dict()["id"]
            self.label_id2name = label_id_mapping.set_index("id").to_dict()["name"]

        if isinstance(scene_ids, str):
            scene_ids = json.load(open(scene_ids, "r"))

        if isinstance(scene_ids, list):
            scene_ids = sorted(scene_ids)
            exist_scenes = set(os.listdir(self.data_dir))
            scene_ids = [
                scene_id for scene_id in scene_ids if scene_id in exist_scenes
            ]  # [:20]
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]
            self.scene_paths = None
        else:
            self.scene_paths = scene_ids  # scene_id -> {img_path, layout_path}
            scene_ids = sorted(list(scene_ids.keys()))  # [:20]
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]

        if depth_data_dir is not None and depth_data_dir.endswith(".json"):
            self.depth_data_dir = json.load(open(depth_data_dir, "r"))
        else:
            self.depth_data_dir = depth_data_dir

        print(f"{mode} Found {len(scene_ids)} scenes")
        graphs = {
            # scene_id: nx.read_gpickle(os.path.join(graph_dir, f"{scene_id}.pkl"))
            scene_id: pickle.load(
                open(os.path.join(graph_dir, f"{scene_id}.pkl"), "rb")
            )
            for scene_id in scene_ids
        }

        # each item_meta contains scene_id and 1 to 3 nodes
        single_node_meta = []
        two_nodes_meta = []
        three_nodes_meta = []
        for scene_id in scene_ids:
            graph = graphs[scene_id]
            cc = [c for c in nx.connected_components(graph)]

            # get single node items
            nodes = sorted(list(graph.nodes))
            single_node_meta.extend([(scene_id, [node]) for node in nodes])

            # get two nodes items
            cc2 = [c for c in cc if len(c) >= 2]
            nodes2 = sum([list(c) for c in cc2], [])
            graph2 = graph.subgraph(nodes2)
            subgraph2 = get_connected_subgraphs(graph2, 2)
            two_nodes_meta.extend([(scene_id, sorted(list(x))) for x in subgraph2])

            # get three nodes items
            cc3 = [c for c in cc if len(c) >= 3]
            nodes3 = sum([list(c) for c in cc3], [])
            graph3 = graph.subgraph(nodes3)
            subgraph3 = get_connected_subgraphs(graph3, 3)
            three_nodes_meta.extend([(scene_id, sorted(list(x))) for x in subgraph3])

        print(f"got {len(single_node_meta)} panos")
        print(f"got {len(two_nodes_meta)} pano pair")
        print(f"got {len(three_nodes_meta)} pano triple")

        # self.graphs = graphs
        # interleave 3 lists
        self.items_meta = list(
            filter(
                lambda i: i is not None,
                chain.from_iterable(
                    zip_longest(single_node_meta, two_nodes_meta, three_nodes_meta)
                ),
            )
        )
        self.items_meta = self.items_meta[::step]
        self._set_K(image_size, cond_image_size, image_strides)
        self._pano_graph = self._make_pano_graph(n_headings, n_elevations)

    def _make_pano_graph(self, n_heading, n_elevation):
        """
        make graph for views within a pano
        for example 3 heading and 2 elevation, the graph is as follow:
            -- 0 -- 1 -- 2 -- 0 --
               |    |    |    |
            -- 3 -- 4 -- 5 -- 3 --
        """
        n_nodes = n_heading * n_elevation
        G = nx.Graph()
        for i in range(n_nodes):
            G.add_node(i)
        for i in range(n_nodes):
            r = i // n_heading
            c = i % n_heading
            G.add_edge(i, ((r + 1) % n_elevation) * n_heading + c)
            G.add_edge(i, r * n_heading + (c + 1) % n_heading)
        return G

    def _get_subgraph_pano(self, n_nodes):
        """
        get subgraph of size n_nodes from the pano graph
        """
        start_node = np.random.choice(list(self._pano_graph.nodes))
        res = []
        recursive_local_expand(
            self._pano_graph,
            {start_node},
            set(self._pano_graph.neighbors(start_node)),
            set(),
            res,
            n_nodes,
            deterministic=False,
            return_one=True,
        )
        return list(res[0])

    def _sample_frames_meta(self, nodes: List[str]):
        """
        sample condition and target from the item_meta

        single pano sample:
            randomly choose subgraph of size 6 from the pano graph

        2 pano sample:
            randomly choose 3 heading (at most 60 deg apart)
            and use these same heading for both pano

        3 pano sample:
            randomly choose 3 heading (at most 60 deg apart)
            the 3 target frames from a single pano
            the 3 condition frames randomly choose from the other 2 panos
            but have same 3 headings

        return frames_meta: {
                cond: [view_id],
                target: [view_id],
            }

        """
        out = {}
        if len(nodes) == 1:
            pano_id = nodes[0]
            if self.mode != "train":
                indices = list(range(6))
                out["cond"] = [f"{pano_id}-{x:0>2}" for x in indices[:3]]
                out["target"] = [f"{pano_id}-{x:0>2}" for x in indices[3:]]
            else:
                indices = np.random.permutation(self._get_subgraph_pano(6))
                meta = [f"{pano_id}-{x:0>2}" for x in indices]
                out["cond"] = meta[:3]
                out["target"] = meta[3:]

        elif len(nodes) == 2:
            if self.mode != "train":
                pano_id1, pano_id2 = nodes
                indices = list(range(3))

                out["cond"] = [f"{pano_id1}-{x:0>2}" for x in indices[:3]]
                out["target"] = [f"{pano_id2}-{x:0>2}" for x in indices[:3]]
            else:
                pano_id1, pano_id2 = np.random.permutation(nodes)
                indices = self._get_subgraph_pano(3)
                out["cond"] = [f"{pano_id1}-{x:0>2}" for x in indices]
                out["target"] = [f"{pano_id2}-{x:0>2}" for x in indices]
                
        elif len(nodes) == 3:
            if self.mode != "train":
                pano_id1, pano_id2, pano_id3 = nodes
                indices = list(range(3))
                cond_pano_ids = [pano_id1, pano_id2, pano_id2]
                out["cond"] = [
                    f"{pano_id}-{x:0>2}"
                    for pano_id, x in zip(cond_pano_ids, indices[:3])
                ]
                out["target"] = [f"{pano_id3}-{x:0>2}" for x in indices[:3]]
            else:
                pano_id1, pano_id2, pano_id3 = np.random.permutation(nodes)
                indices = np.random.permutation(self._get_subgraph_pano(3))
                cond_pano_ids = [pano_id1, pano_id2, pano_id2]
                out["cond"] = [
                    f"{pano_id}-{x:0>2}" for pano_id, x in zip(cond_pano_ids, indices)
                ]
                out["target"] = [f"{pano_id3}-{x:0>2}" for x in indices]
        return out

    def __getitem__(self, index):
        scene_id, nodes = self.items_meta[index]
        frames_meta = self._sample_frames_meta(nodes)
        data = self.get_item_from_meta(scene_id, frames_meta)
        if self.scene_paths is None:
            layout_dir = self.layout_dir
        else:
            layout_dir = self.scene_paths[scene_id]["layout_path"]
        data.update(
            self.get_layout(
                scene_id=scene_id,
                frame_ids=frames_meta["target"],
                poses=data["pose_out"],
                layout_dir=layout_dir,
            )
        )
        return data