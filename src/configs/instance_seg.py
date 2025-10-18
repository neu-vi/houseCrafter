"""
make seg model



make metadata for the evaluation set (2d) (loop through dataset)

make metadata for the evaluation set (3d) (grouping by room, remove wall only frame)
    load from pose files (apply the same filtering as the rendering to get the gt images)and house 

make dataloader for generated images (based on metadata) (can resuse odin code, change the way to load data)
    2d load from generated images 
    3d rendering from mesh on the fly

make dataloader for gt images based on metadata
    2d load from gt images DONE
    3d load from gt images

prediction post processing (extract 2d boxes, classes for each instance) DONE

-----------------
2d extract gt boxes (camera frustum vs box corners) and in the same room

3d extract gt boxes (in each room)
--------------
"""

import argparse
import torch
import os
from torch.utils.data import Dataset, DataLoader
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from odin.data_video.dataset_mapper_front3d import Front3DDatasetMapper, DB
from odin.data_video.build import collate_fn
from odin import add_maskformer2_video_config, add_maskformer2_config
from odin.config import add_front3d_config
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import json
from detectron2.config import get_cfg
import copy
from odin.data_video.dataset_mapper_front3d_2 import Front3DDatasetMapper2
from accelerate import Accelerator
from omegaconf import OmegaConf
import pickle
from tqdm import tqdm


def make_model(args):
    det_cfg = get_cfg()
    add_deeplab_config(det_cfg)
    add_maskformer2_config(det_cfg)
    add_maskformer2_video_config(det_cfg)
    add_front3d_config(det_cfg)
    # TODO add way to get config path
    det_cfg.merge_from_file("./configs/scannet_context/swin_3d.yaml")
    # det_cfg.merge_from_file("./configs/scannet_context/front3d_neu.yaml")
    det_cfg.merge_from_file("./configs/scannet_context/front3d_neu256.yaml")
    # det_cfg.merge_from_file("./scannet_swin.yaml")

    det_cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    det_cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    det_cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    det_cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8

    ### custom
    det_cfg.INPUT.STRONG_AUGS = False
    det_cfg.USE_GHOST_POINTS = False
    det_cfg.HIGH_RES_INPUT = True
    det_cfg.MODEL.SUPERVISE_SPARSE = False
    det_cfg.TEST.EVAL_SPARSE = False
    det_cfg.USE_SEGMENTS = False
    det_cfg.TEST_CLASS_AGNOSTIC = False
    det_cfg.SKIP_CLASSES = None
    det_cfg.MODEL.WEIGHTS = args.ckpt  # "/mnt/Data/hieu/odin/size512_29k.pth"
    # det_cfg.INPUT.VOXELIZE = False
    det_cfg.freeze()
    model = build_model(det_cfg)
    DetectionCheckpointer(model, save_dir=det_cfg.OUTPUT_DIR).resume_or_load(
        det_cfg.MODEL.WEIGHTS, resume=False
    )
    model.eval()

    # metadata = MetadataCatalog.get(
    #     det_cfg.DATASETS.TEST[0] if len(det_cfg.DATASETS.TEST) else "__unused"
    # )
    return model, det_cfg


class GTDataset(Dataset):
    IMAGE_SIZE = 256

    def __init__(self, det_cfg, data_cfg):
        self.data_cfg = data_cfg
        self.det_cfg = det_cfg

        # list of {scene_id, image_ids, item_id}
        # item id can be start_node for 2d dataset
        # or room_id for 3d dataset
        self.metadata = json.load(open(data_cfg.metadata))
        self.mapper = Front3DDatasetMapper(det_cfg, is_test=True, dataset_name="")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.metadata[idx])
        data["height"] = self.IMAGE_SIZE
        data["width"] = self.IMAGE_SIZE
        data["img_db_path"] = os.path.join(self.data_cfg.img_path, data["scene_id"])
        data = self.mapper(data)
        return data


# def make_gt_dataloader(det_cfg, data_cfg):
#     dataset = GTDataset(det_cfg, data_cfg)
#     dataloader = DataLoader(
#         dataset, collate_fn=lambda x: x, batch_size=1, shuffle=False
#     )
#     return dataloader


class PostProcessor:
    def __init__(self, class_mapping={}, score_threshold=0.7, size_cutoff=0.05):
        """
        class mapping: id->name

        """
        self.score_threshold = score_threshold
        self.class_mapping = class_mapping
        self.size_cutoff = (
            size_cutoff  # cutoff x mean take the percentile from x to 1-x
        )

    def __call__(self, input_dict, output_dict):
        """
        input_dict: input dataset dict
        output_dict: output dict from mdoel
        return list of {score, class_id, bbox: xzxz}
        """
        pred_scores = output_dict["instances_3d"]["pred_scores"]
        pred_masks = output_dict["instances_3d"]["pred_masks"]
        pred_labels = output_dict["instances_3d"]["pred_classes"]

        sort_idx = torch.argsort(pred_scores)
        pred_masks = pred_masks.permute(1, 0)[sort_idx]
        pred_labels = pred_labels[sort_idx]

        # select confident predictions
        pred_scores = pred_scores[sort_idx]

        conf = pred_scores > self.score_threshold
        pred_masks = pred_masks[conf]
        pred_labels = pred_labels[conf]

        # collapse masks
        single_masks = pred_masks * (torch.arange(pred_masks.shape[0])[:, None] + 1)
        single_masks = single_masks.argmax(dim=0)

        our_pc = (
            F.interpolate(
                input_dict["original_xyz"].float().permute(0, 3, 1, 2),
                scale_factor=0.5,
                mode="nearest",
            )
            .permute(0, 2, 3, 1)
            .reshape(-1, 3)
        )
        bboxes = self._get_bounding_box(single_masks, our_pc, pred_masks.shape[0])

        out = []
        for label, bbox, score in zip(pred_labels, bboxes, pred_scores):
            if bbox is None:
                continue
            out.append({"score": float(score), "class_id": int(label), "bbox": bbox})

    def _get_bounding_box(self, masks, pcd, max_index):
        """
        get bbox in xz, ignore y
        mask: (N)
        pcd: (N,3)

        return bbox: xzxz
        """
        xz_pcd = pcd[:, [0, 2]]
        bboxes = []
        for index in range(1, max_index + 1):
            mask = masks == index
            if mask.sum() == 0:
                bboxes.append(None)
                continue

            instance_pcd = xz_pcd[mask]  # (NN,2)
            lower_bound = torch.quantile(
                instance_pcd, self.size_cutoff, dim=0, interpolation="lower"
            )
            upper_bound = torch.quantile(
                instance_pcd, 1 - self.size_cutoff, dim=0, interpolation="higher"
            )
            bboxes.append(
                map(
                    float,
                    (lower_bound[0], lower_bound[1], upper_bound[0], upper_bound[1]),
                )
            )
        return bboxes


class GenDataset(Dataset):
    """
    can be used for both 2d and 3d case, assume rendered images from3d are saved in the same way as
    2d images
    """

    IMAGE_SIZE = 256

    def __init__(self, det_cfg, data_cfg):
        self.data_cfg = data_cfg
        self.det_cfg = det_cfg

        # list of {scene_id, image_ids, item_id}
        # item id can be start_node for 2d dataset
        # or room_id for 3d dataset
        self.metadata = json.load(open(data_cfg.metadata))
        self.mapper = Front3DDatasetMapper(det_cfg, is_test=True, dataset_name="")

    def __getitem__(self, idx):
        data = copy.deepcopy(self.metadata[idx])
        data["height"] = self.IMAGE_SIZE
        data["width"] = self.IMAGE_SIZE
        # this is to load pose
        data["gt_img_db_path"] = os.path.join(self.data_cfg.img_path, data["scene_id"])
        # this is to load genearated rgbd
        data["gen_img_db_path"] = os.path.join(
            self.data_cfg.gen_img_path, data["scene_id"]
        )
        data = self.mapper(data)
        return data


class Front3DGenDatasetMapper2(Front3DDatasetMapper2):
    """
    can be used for both 2d and 3d case, assume rendered images from3d are saved in the same way as
    2d images
    """

    def load_rgbdpose(self, dataset_dict, frame_ids):
        gt_img_db = DB(dataset_dict["gt_img_db_path"])
        images, depths, poses = [], [], []
        # TODO: load generated images
        for frame_id in frame_ids:
            image = img_db.get_rgb(frame_id)
            depth = img_db.get_depth(frame_id)
            pose = gt_img_db.get_pose(frame_id)
            images.append(image)
            depths.append(depth)
            poses.append(pose)
        return images, depths, poses


def main_gt(args, data_cfg):
    """
    run instance seg on gt dataset
    """
    model, det_cfg = make_model()
    dataset = GTDataset(det_cfg, data_cfg)
    dataloader = DataLoader(
        dataset, collate_fn=lambda x: x, batch_size=1, shuffle=False
    )
    accelerator = Accelerator()

    dataloader = accelerator.prepare(dataloader)
    model = model.to(accelerator.device)
    post_processor = PostProcessor(
        size_cutoff=data_cfg.size_cutoff, score_threshold=data_cfg.score_threhold
    )
    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    os.makedirs(args.outdir, exist_ok=True)
    for data in dataloader:
        item_id = data[0]["item_id"]
        out_file = os.path.join(args.outdir, f"{item_id}.pkl")
        if os.path.exists(out_file):
            progress_bar.update(1)
            continue
        with torch.no_grad():
            output_dict = model(data)
            output = post_processor(data[0], output_dict[0])
        pickle.dump(output, open(out_file, "wb"))
        progress_bar.update(1)


def main_gen2d(args, data_cfg):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--outdir", type=str)

    args, unknown = parser.parse_known_args()
    data_cfg = OmegaConf.load(args.data_cfg)
    cli = OmegaConf.from_dotlist(unknown)
    data_cfg = OmegaConf.merge(data_cfg, cli)

    main_gt(args, data_cfg)
