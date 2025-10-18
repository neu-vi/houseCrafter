import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from cfg_util import get_obj_from_str, instantiate_from_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .data_utils import collate_fn
from .front3d import Front3DPose


class ConcatLayoutRCNFront3DDataModule:
    def __init__(
        self,
        image_strides: List[int],
        batch_size,
        graph_dir,
        train_dir,  #######
        train_scene_ids,  ########
        val_scene_ids,  ###########
        dataset_cls: str,  ##########
        pose_dir=None,  ##########
        location_dir=None,
        val_dir=None,  #############
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
        layout_dir: str = "",  #############
        max_num_points=20,
        intersection_pos="height",
        dataset_ratio=None,
        dataset_size=10_000_000,
        train_depth_dir=None,
        val_depth_dir=None,
        enable_remapping_cls=False,
        layout_label_ids="",
        **kwargs,
    ):

        self.kwargs = kwargs
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
        self.dataset_ratio = dataset_ratio
        self.dataset_size = dataset_size

        self.enable_remapping_cls = enable_remapping_cls
        self.layout_label_ids = layout_label_ids

        self.train_scene_ids = [json.load(open(x, "r")) for x in train_scene_ids]
        self.val_scene_ids = [json.load(open(x, "r")) for x in val_scene_ids]

        if train_depth_dir is None:
            self.train_depth_dir = [None] * len(train_dir)
        else:
            self.train_depth_dir = train_depth_dir

        if val_depth_dir is None:
            val_depth_dir = [None] * len(val_dir)
        else:
            self.val_depth_dir = val_depth_dir
        # layout args
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos

        for _dataset in dataset_cls:
            assert _dataset.split(".")[-1] in [
                "Front3DPose",
                "Front3DLocation",
                "Front3DObjPose",
            ]
        self.tform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        if load_target_depth:
            self.o_depth_tform = transforms.Compose(
                [
                    lambda x: torch.tensor(x, dtype=torch.float32),
                    instantiate_from_config(depth_tform_cfg),
                ]
            )
        else:
            self.o_depth_tform = None

    def train_dataloader(self):
        datasets = [
            get_obj_from_str(_dataset_cls)(
                image_strides=self.image_strides,
                graph_dir=self.graph_dir,
                root_dir=_train_dir,
                scene_ids=_train_scene_ids,
                pose_dir=_pose_dir,
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
                layout_dir=_layout_dir,
                max_num_points=self.max_num_points,
                intersection_pos=self.intersection_pos,
                depth_data_dir=_train_depth_dir,
                enable_remapping_cls=self.enable_remapping_cls,
                layout_label_ids=self.layout_label_ids,
                **self.kwargs,
            )
            for (
                _train_dir,
                _train_scene_ids,
                _dataset_cls,
                _pose_dir,
                _layout_dir,
                _train_depth_dir,
            ) in zip(
                self.train_dir,
                self.train_scene_ids,
                self.dataset_cls,
                self.pose_dir,
                self.layout_dir,
                self.train_depth_dir,
            )
        ]
        dataset = ConcatDataset(datasets, self.dataset_ratio, self.dataset_size)

        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def train_log_dataloader(self):
        datasets = [
            get_obj_from_str(_dataset_cls)(
                image_strides=self.image_strides,
                graph_dir=self.graph_dir,
                root_dir=_train_dir,
                scene_ids=_train_scene_ids,
                pose_dir=_pose_dir,
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
                layout_dir=_layout_dir,
                max_num_points=self.max_num_points,
                intersection_pos=self.intersection_pos,
                depth_data_dir=_train_depth_dir,
                enable_remapping_cls=self.enable_remapping_cls,
                layout_label_ids=self.layout_label_ids,
                **self.kwargs,
            )
            for (
                _train_dir,
                _train_scene_ids,
                _dataset_cls,
                _pose_dir,
                _layout_dir,
                _train_depth_dir,
            ) in zip(
                self.train_dir,
                self.train_scene_ids,
                self.dataset_cls,
                self.pose_dir,
                self.layout_dir,
                self.train_depth_dir,
            )
        ]
        dataset = ConcatDataset(datasets, None)

        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            collate_fn=collate_fn,
        )

    def val_dataloader(self, step=1, n_workers=1):
        datasets = [
            get_obj_from_str(_dataset_cls)(
                image_strides=self.image_strides,
                graph_dir=self.graph_dir,
                root_dir=_val_dir,
                scene_ids=_val_scene_ids,
                pose_dir=_pose_dir,
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
                layout_dir=_layout_dir,
                max_num_points=self.max_num_points,
                intersection_pos=self.intersection_pos,
                step=step,
                depth_data_dir=_val_depth_dir,
                enable_remapping_cls=self.enable_remapping_cls,
                layout_label_ids=self.layout_label_ids,
                **self.kwargs,
            )
            for (
                _val_scene_ids,
                _dataset_cls,
                _pose_dir,
                _val_dir,
                _layout_dir,
                _val_depth_dir,
            ) in zip(
                self.val_scene_ids,
                self.dataset_cls,
                self.pose_dir,
                self.val_dir,
                self.layout_dir,
                self.val_depth_dir,
            )
        ]
        dataset = ConcatDataset(datasets, None)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=n_workers,
            collate_fn=collate_fn,
        )


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset], ratio=None, dataset_size=100_000_000):
        """
        ratio: giving the prob of getting sample from corresponding dataset

        """
        self.sizes = np.array([len(dataset) for dataset in datasets])
        if ratio is None:
            self._offset = np.cumsum(self.sizes)
            self._len = self._offset[-1]
        else:
            ratio = np.array(ratio) / sum(ratio)
            self._offset = np.cumsum(ratio) * dataset_size
            self._len = dataset_size
        self._datasets = datasets

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        dataset_id = np.searchsorted(self._offset, index)
        dataset = self._datasets[dataset_id]
        dataset_index = int(self._offset[dataset_id] - index) % len(dataset)
        return dataset[dataset_index]


class Front3DObjPose(Front3DPose):
    def __init__(
        self,
        image_strides: List[int],
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
        self.return_depth_input = return_depth_input
        # layout var
        self.layout_shape_db = layout_shape_db
        self.layout_dir = layout_dir
        self.max_num_points = max_num_points
        self.intersection_pos = intersection_pos
        self.enable_remapping_cls = enable_remapping_cls
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
            scene_ids = [scene_id for scene_id in scene_ids if scene_id in exist_scenes]
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]
            self.scene_paths = None
        else:
            self.scene_paths = scene_ids  # scene_id -> {img_path, layout_path}
            scene_ids = sorted(list(scene_ids.keys()))
            if n_scenes is not None:
                scene_ids = scene_ids[:n_scenes]
                
        self.depth_data_dir = depth_data_dir

        print(f"{mode} Found {len(scene_ids)} scenes")

        # each item is object, in training, sampling 3 cond and 3 target
        total_frames = n_cond_frames + n_target_frames
        items_meta = []
        all_pose_ids = defaultdict(dict)
        for scene_id in scene_ids:
            pose_scene_dir = os.path.join(pose_dir, scene_id)
            objs = sorted(os.listdir(pose_scene_dir))
            for obj_file in objs:
                obj_id = obj_file.replace(".json", "")
                pose_ids = json.load(open(os.path.join(pose_scene_dir, obj_file), "r"))
                if len(pose_ids) > total_frames:
                    all_pose_ids[scene_id][obj_id] = sorted(pose_ids)
                    items_meta.append((scene_id, obj_id))

        self.items_meta = items_meta[::step]
        self.all_pose_ids = all_pose_ids
        self._set_K(image_size, cond_image_size, image_strides)

    def __getitem__(self, index):
        scene_id, object_id = self.items_meta[index]
        frames_meta = self._sample_frames_meta(scene_id, object_id)
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
            data["start_node"] = object_id
        return data

    def _sample_frames_meta(
        self, scene_id: str, object_id: str
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
        frame_ids = self.all_pose_ids[scene_id][object_id]
        if self.mode == "train":
            frame_ids = np.random.permutation(frame_ids).tolist()
        return {
            "cond": frame_ids[: self.n_cond_frames],
            "target": frame_ids[self.n_cond_frames : total_frames],
        }
