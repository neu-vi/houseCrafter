import copy
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
from .front3d_combine import ConcatDataset


class ConcatLayoutRCNFront3DDataModule:
    def __init__(
        self,
        image_strides: List[int],
        batch_size,
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
        depth_tform_cfg: Dict = None,
        # layout args
        layout_shape_db=(512, 512),
        max_num_points=20,
        intersection_pos="height",
        dataset_ratio=None,
        dataset_size=10_000_000,
        enable_remapping_cls=False,
        layout_label_ids="",
        datasets_cfg=[],
        **kwargs,
    ):
        common_args = kwargs
        common_args["image_size"] = image_size
        common_args["image_strides"] = image_strides
        common_args["cond_image_size"] = cond_image_size
        common_args["cond_image_strides"] = cond_image_strides
        common_args["n_cond_frames"] = n_cond_frames
        common_args["n_target_frames"] = n_target_frames
        common_args["depth_shape_db"] = depth_shape_db
        common_args["load_target_depth"] = load_target_depth
        common_args["load_cond_depth"] = load_cond_depth
        common_args["return_depth_input"] = return_depth_input
        common_args["depth_scale"] = depth_scale
        common_args["layout_shape_db"] = layout_shape_db
        common_args["max_num_points"] = max_num_points
        common_args["intersection_pos"] = intersection_pos
        common_args["enable_remapping_cls"] = enable_remapping_cls
        common_args["layout_label_ids"] = layout_label_ids
        common_args["image_transforms"] = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        if load_target_depth:
            o_depth_tform = transforms.Compose(
                [
                    lambda x: torch.tensor(x, dtype=torch.float32),
                    instantiate_from_config(depth_tform_cfg),
                ]
            )
        else:
            o_depth_tform = None
        common_args["o_depth_tform"] = o_depth_tform
        
            
        self.common_args = common_args
        self.datasets_cfg = datasets_cfg
        self.dataset_ratio = dataset_ratio
        self.dataset_size = dataset_size
        self.num_workers = num_workers
        self.batch_size = batch_size
            
    def train_dataloader(self):
        datasets = []
        for dataset_cfg in self.datasets_cfg:
            cfg = copy.deepcopy(self.common_args)
            cfg.update(dataset_cfg["params"])
            cfg["root_dir"] = cfg.get("train_dir", None)
            cfg["scene_ids"] = cfg.get("train_scene_ids", None)
            cfg["mode"] = "train"
            cfg["depth_data_dir"] = cfg.get("train_depth_dir", None)
            datasets.append(get_obj_from_str(dataset_cfg["target"])(**cfg))
        dataset = ConcatDataset(datasets, self.dataset_ratio, self.dataset_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
    
    def train_log_dataloader(self):
        datasets = []
        for dataset_cfg in self.datasets_cfg:
            cfg = copy.deepcopy(self.common_args)
            cfg.update(dataset_cfg["params"])
            cfg["root_dir"] = cfg.get("train_dir", None)
            cfg["scene_ids"] = cfg.get("train_scene_ids", None)
            cfg["mode"] = "val"
            cfg["depth_data_dir"] = cfg.get("train_depth_dir", None)
            datasets.append(get_obj_from_str(dataset_cfg["target"])(**cfg))
        dataset = ConcatDataset(datasets, None)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self, step=1, n_workers=1):
        datasets = []
        for dataset_cfg in self.datasets_cfg:
            cfg = copy.deepcopy(self.common_args)
            cfg.update(dataset_cfg["params"])
            cfg["root_dir"] = cfg.get("val_dir", None)
            cfg["scene_ids"] = cfg.get("val_scene_ids", None)
            cfg["mode"] = "val"
            cfg["depth_data_dir"] = cfg.get("val_depth_dir", None)
            cfg["step"] = step
            datasets.append(get_obj_from_str(dataset_cfg["target"])(**cfg))
        dataset = ConcatDataset(datasets, None)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_fn,
        )