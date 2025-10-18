"""
############# refactored ###########

init model
load graph

write dataloading function
write data preprocessing 
write graph traverse class:
 breath first search

select cond and target:
    from an unvisited node find all node within 2 hop distance
    unvisited one-hope node is the target
    visted one-hop and two-hop node is the cond
"""

import sys
from collections import deque

import numpy as np

sys.path.append("/work/vig/hieu/escher/6DoF")
import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from CN_encoder import CN_encoder
from diffusers import AutoencoderKL, DDIMInverseScheduler, DDIMScheduler
from einops import rearrange, repeat
from skimage.metrics import structural_similarity as calculate_ssim
from torchvision import transforms
from tqdm import tqdm

# from train_eschernet_scannet import log_validation, parse_args
from unet_2d_condition import UNet2DConditionModel

LPIPS = lpips.LPIPS(net="alex", version="0.1")
import argparse
import json
import os
import zlib

import lmdb
import matplotlib.pyplot as plt
import networkx as nx
from cfg_util import instantiate_from_config
from data_modules.data_utils import (
    Torch3DRenderer,
    collate_fn,
    get_absolute_depth,
    render_target_view,
    render_target_view_torch3d,
)
from data_modules.front3d import Front3DPose
from data_modules.front3d_layout_obj import Front3DPoseObjLayout
from generation_utils import (
    GraphSearch2,
    generate_ddim_inversion,
    make_pipeline,
    preprocess_image,
)
from omegaconf import OmegaConf
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline

MIN_DEPTH = 0.1  # m


# init model and dataset
def show_img(img):
    plt.close()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def save_img(img, path):
    plt.close()
    # plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)


def preprocess_image(img):
    """
    return
        img in np.uint8 [0,255]
        img in torch.float32 [0,255]
    """
    img = (img + 1) / 2.0 * 255.0
    img = torch.clamp(img, min=0.0, max=255.0)
    np_img = img.cpu().numpy()
    np_img = np.round(np_img).astype(np.uint8)
    return img, np_img


def process_depth(depth, max_depth=30.0):
    """
    args:
        depth in torch.float32 [-1,1] (t h w 3)
    return
        depth in np.uint16 [0,2**15] (t h w) in mm
        depth in torch.float32  (t h w) in meter
    """
    depth = depth.mean(-1)
    depth = (depth + 1) * (0.5 * max_depth)
    np_depth = (1000.0 * depth).cpu().numpy()
    np_depth = np.round(np_depth).astype(np.uint16)
    return depth, np_depth


def process_calib_depth(depth):
    """
    args:
        depth torch.float32 in meter (t h w)
    return
        a_depth in np.uint16 in mm (t h w)
    """
    depth = depth * 1000
    depth[depth >= 2**16] = 0
    depth = depth.cpu().numpy()
    depth = np.round(depth).astype(np.uint16)
    return depth


def process_output(output, max_depth=30.0):
    output = rearrange(output, "(t n) c h w -> n t h w c", n=2)
    img, np_img = preprocess_image(output[0])
    depth, np_depth = process_depth(output[1], max_depth)
    return img, depth, np_img, np_depth  # t h w c and t h w


def save_rgbdd(
    out_dir,
    frames_meta,
    imgs,
    calib_depths,
    absolute_depths,
    db,
    warp_imgs=None,
    save_img=True,
):
    if save_img:
        os.makedirs(f"{out_dir}/img", exist_ok=True)
        # os.makedirs(f"{out_dir}/relative_depth/{scene_id}", exist_ok=True)
        os.makedirs(f"{out_dir}/absolute_depth", exist_ok=True)
        os.makedirs(f"{out_dir}/warp_img", exist_ok=True)
    if warp_imgs is None:
        warp_imgs = [None] * len(imgs)
    if calib_depths is None:
        calib_depths = [None] * len(imgs)
    out_txn = db.begin(write=True)
    for img, a_depth, c_depth, frame_id, warp_img in zip(
        imgs, absolute_depths, calib_depths, frames_meta, warp_imgs
    ):
        if save_img:
            img_path = f"{out_dir}/img/{frame_id}.jpeg"
            # r_depth_path = f"{out_dir}/relative_depth/{scene_id}/{frame_id}.png"
            a_depth_path = f"{out_dir}/absolute_depth/{frame_id}.png"
            plt.imsave(img_path, img)
            # cv2.imwrite(r_depth_path, r_depth)
            cv2.imwrite(a_depth_path, a_depth)

            if warp_img is not None:
                warp_img_path = f"{out_dir}/warp_img/{frame_id}.jpeg"
                plt.imsave(warp_img_path, warp_img)

        rgb_key = f"{frame_id}_rgb".encode("ascii")
        img = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        out_txn.put(rgb_key, img)

        a_depth_key = f"{frame_id}_depth".encode("ascii")
        a_depth = zlib.compress(a_depth.tobytes())
        out_txn.put(a_depth_key, a_depth)

        # c_depth_key = f"{frame_id}_calibdepth".encode("ascii")
        # c_depth = zlib.compress(c_depth.tobytes())
        # out_txn.put(c_depth_key, c_depth)
    out_txn.commit()


def make_dataset(cfg, out_dir_db, scene_id):
    data_kw = OmegaConf.to_container(cfg.data.params)

    data_kw["target_dir"] = data_kw["val_dir"]

    for key in ["train_dir", "val_dir", "dataset_cls", "batch_size", "num_workers"]:
        data_kw.pop(key)
    tform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    if "depth_tform_cfg" in data_kw:
        o_depth_tform = transforms.Compose(
            [
                lambda x: torch.tensor(x, dtype=torch.float32),
                instantiate_from_config(data_kw["depth_tform_cfg"]),
            ]
        )
        data_kw["o_depth_tform"] = o_depth_tform
    data_kw["root_dir"] = out_dir_db
    data_kw["image_transforms"] = tform

    # scene_id = val_scene_ids[0] #"031fba32-7c48-4c1e-8342-aab66d6e531f"
    data_kw["scene_ids"] = [scene_id]
    out_db_dir = os.path.join(out_dir_db, scene_id)
    os.makedirs(out_db_dir, exist_ok=True)
    dataset = Front3DPose(**data_kw)
    return dataset

def make_dataset2(cfg, out_dir_db, scene_id):
    # todo: understand how the data loader gets all the graph, layout, pose, and image, especially how to swap out the image
    data_kw = OmegaConf.to_container(cfg.data.params)
    data_kw2 = OmegaConf.to_container(cfg.data.params.datasets_cfg[0].params)
    data_kw.update(data_kw2)

    data_kw["val_dir"] = cfg.data.params.val_dir
    data_kw["layout_dir"] = cfg.data.params.layout_dir
    data_kw["val_scene_ids"] = cfg.data.params.val_scene_ids
    # inspection codes, delete later
    print('-----args in make_dataset2 data_kw-----')
    print('validation dir:',data_kw["val_dir"])
    print('layout dir:',data_kw["layout_dir"])
    print('val scene ids:',data_kw["val_scene_ids"])

    # todo: figure out why the setting is like this
    data_kw["target_dir"] = data_kw["val_dir"]

    for key in ["train_dir", "val_dir", "dataset_cls", "batch_size", "num_workers"]:
        data_kw.pop(key)
    tform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    if "depth_tform_cfg" in data_kw:
        o_depth_tform = transforms.Compose(
            [
                lambda x: torch.tensor(x, dtype=torch.float32),
                instantiate_from_config(data_kw["depth_tform_cfg"]),
            ]
        )
        data_kw["o_depth_tform"] = o_depth_tform
    data_kw["root_dir"] = out_dir_db
    data_kw["image_transforms"] = tform

    # scene_id = val_scene_ids[0] #"031fba32-7c48-4c1e-8342-aab66d6e531f"
    data_kw["scene_ids"] = [scene_id]
    out_db_dir = os.path.join(out_dir_db, scene_id)
    os.makedirs(out_db_dir, exist_ok=True)
    # todo: figure out what is the difference between Front3DPose and Front3DPoseObjLayout
    dataset = Front3DPoseObjLayout(**data_kw)
    return dataset

def is_in_box(pose):
    # filter pose in the box
    # xyzxyz
    # living room
    # "031fba32-7c48-4c1e-8342-aab66d6e531f"
    # box = np.array(
    #     [
    #         -3.828900098800659,
    #         0.0,
    #         -2.2542001008987427,
    #         -0.05559999495744705,
    #         2.5999999046325684,
    #         3.165000081062317,
    #     ]
    # )
    # master bedroom
    # "031fba32-7c48-4c1e-8342-aab66d6e531f"
    # box = np.array(
    #     [
    #         -0.1756000518798828,
    #         0.0,
    #         -5.403500080108643,
    #         4.7505998611450195,
    #         2.5999999046325684,
    #         -2.0141998529434204,
    #     ]
    # )
    # xyz = pose[:3, 3]
    # return np.all(np.logical_and(xyz >= box[:3], xyz <= box[3:]))

    # run on the whole scene
    return True


def setup_out_db(dataset, scene_id):
    """
    copy cond frame from src_db to des_db
    copy pose from src_db to des_db
    may filter the pose to be in a room
    # make dummy frame for cond view
    """
    src_db = lmdb.open(
        os.path.join(dataset.target_dir, scene_id),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    out_db_dir = os.path.join(dataset.data_dir, scene_id)
    os.makedirs(out_db_dir, exist_ok=True)
    des_db = lmdb.open(out_db_dir, map_size=int(1e12))

    poses = {}
    in_key = []

    with src_db.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode() for key in exist_keys]
        exist_keys = [key for key in exist_keys if key.endswith("_pose")]
        print(len(exist_keys))
        for key in exist_keys:
            pose = txn.get(key.encode("ascii"))
            pose = (
                np.frombuffer(zlib.decompress(pose), dtype=np.float32)
                .reshape(4, 4)
                .copy()
            )
            pose[3, 3] = 1.0
            poses[key] = pose
            if is_in_box(pose):
                in_key.append(key.replace("_pose", ""))

    graph = dataset.graphs[scene_id].subgraph(in_key)
    cc = sorted(
        [c for c in nx.connected_components(graph)], key=lambda x: len(x), reverse=True
    )
    graph = graph.subgraph(cc[0]).copy()

    # cond view for bedroom
    # "031fba32-7c48-4c1e-8342-aab66d6e531f"
    # cond_view = sorted(in_key)[0]

    # cond view for living room
    # "031fba32-7c48-4c1e-8342-aab66d6e531f"
    # cond_view = sorted(in_key)[-5]

    cond_view = sorted(cc[0])[0]

    make_dummy_view(src_db, des_db, cond_view)
    copy_pose(src_db, des_db, in_key)
    return graph, cond_view, des_db


def make_dummy_view(in_db, out_db, view):
    out_txn = out_db.begin(write=True)
    with in_db.begin() as in_txn:
        rgb_key = f"{view}_rgb".encode("ascii")
        rgb = in_txn.get(rgb_key)

        # decode
        rgb = np.frombuffer(rgb, dtype=np.uint8)
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        dummy_rgb = np.zeros_like(rgb)
        dummy_rgb = cv2.imencode(".jpg", dummy_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        out_txn.put(rgb_key, dummy_rgb)

        pose_key = f"{view}_pose".encode("ascii")
        pose = in_txn.get(pose_key)
        print(rgb is None)
        out_txn.put(pose_key, pose)

        depth_key = f"{view}_depth".encode("ascii")
        depth = in_txn.get(depth_key)
        depth = zlib.decompress(depth)
        depth = np.frombuffer(depth, dtype=np.uint16)
        size = int(np.sqrt(len(depth)))
        depth = depth.reshape(size, size)
        dummy_depth = np.zeros_like(depth)
        dummy_depth[:10, :10] = 1000
        dummy_depth = zlib.compress(dummy_depth.tobytes())
        out_txn.put(depth_key, dummy_depth)
    out_txn.commit()


def copy_pose(in_db, out_db, views):
    out_txn = out_db.begin(write=True)
    with in_db.begin() as in_txn:
        for view in views:
            pose_key = f"{view}_pose".encode("ascii")
            pose = in_txn.get(pose_key)
            out_txn.put(pose_key, pose)
    out_txn.commit()


def main_ddim_inversion(
    cfg,
    output_dir,
    scene_id,
    cond_hop,
    target_hop,
    device,
    pipeline,
    depth_model,
    torch_renderer,
    max_view=1000,
    weight_dtype=torch.float16,
    cache_img=False,
    ddim_inversion=True,
    gaussian_clip=0.0,
):

    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    # dataset = make_dataset(cfg, os.path.join(output_dir, "db"), scene_id)
    # inspection code, delete later:
    print('-----args in main_ddim_inversion-----')
    print('validation dir:',cfg.data.params.val_dir)
    print('layout dir:',cfg.data.params.layout_dir)
    print('val scene ids:',cfg.data.params.val_scene_ids)

    # todo: figure out the behavioral difference between make_dataset and make_dataset2
    dataset = make_dataset2(cfg, os.path.join(output_dir, "db"), scene_id)
    # todo: generally loads the graph, conditional selection, and the target dir for inference batches, figure out the details later
    graph, cond_view, des_db = setup_out_db(dataset, scene_id)
    keys = set(list(graph.nodes))
    if check_compeleted(os.path.join(output_dir, "db", scene_id), keys):
        print(f"scene {scene_id} is already completed")
        return
    depth_func = lambda x: depth_model.infer(x[None, ...])["depth"][0, 0]
    depth_transform = instantiate_from_config(cfg.data.params.depth_tform_cfg)
    i = 0
    # current_node = graph_search.next_node()
    # special choice of condition and target for the first node
    # NOTE either chose graph search 1 or 2
    # graph_search = GraphSearch(graph, cond_view)
    # grpah_search: util class for manaing the graph
    graph_search = GraphSearch2(
        graph, cond_view, cond_hop=cond_hop, target_hop=target_hop
    )
    max_depth = cfg.data.params.depth_tform_cfg.params.max_depth
    progress_bar = tqdm(total=len(graph.nodes))
    sequence = []
    # inspection code, delete later
    print('-----args about the data loading-----')
    print('dataset data dir:',dataset.data_dir)
    print('dataset target dir:',dataset.target_dir)
    print('dataset set scene_path:',dataset.scene_paths)
    while True:
        torch.cuda.empty_cache()
        target_node = graph_search.next_node()
        if target_node is None:
            break
        cond, target = graph_search.get_cond_target(target_node)
        if i == 0:
            assert len(cond) == 0
            assert cond_view in target
            cond = [cond_view]
        target = target[:max_view]
        for n in target:
            graph_search.set_visited(n)
        frames_meta = {
            "cond": cond,
            "target": target,
        }
        # todo: figure out if the cond was swapped out here, or in the next function
        frames_meta['cond'] = [cond_view]
        batch = dataset.get_item_from_meta(scene_id, frames_meta,gt_for_reference=True)
        batch.update(
            dataset.get_layout(
                scene_id=scene_id,
                frame_ids=frames_meta["target"],
                poses=batch["pose_out"],
                layout_dir=dataset.layout_dir,
            )
        )
        batch = collate_fn([batch])
        pred_imgs, warp_imgs = generate_ddim_inversion(
            pipeline,
            batch,
            weight_dtype,
            generator,
            device,
            depth_transform,
            use_ray=True,
            output_depth=True,
            torch_renderer=torch_renderer,
            ignore_prompts=i == 0,
            ddim_inversion=ddim_inversion,
            gaussian_clip=gaussian_clip,
        )
        imgs, a_depths, np_imgs, np_a_depths = process_output(
            pred_imgs, max_depth=max_depth
        )
        # calib_depths, methods = get_absolute_depth(
        #     imgs=rearrange(imgs, "t h w c -> t c h w"),
        #     relative_depths=a_depths,
        #     frames_meta=frames_meta["target"],
        #     depth_model=depth_func,
        #     layout_db_path=os.path.join(dataset.layout_dir, scene_id),
        # )
        # sequence.append({"frame_ids": frames_meta, "depth_method": methods})
        sequence.append({"frame_ids": frames_meta, "depth_method": "none"})
        invalid_a_depth_mask = (a_depths > (max_depth - 0.2)) | (a_depths < MIN_DEPTH)
        a_depths[invalid_a_depth_mask] = 0.0
        # np_calib_depths = process_calib_depth(calib_depths)
        if warp_imgs is not None:
            warp_imgs = rearrange(warp_imgs, "1 t c h w -> t h w c")
            _, warp_imgs = preprocess_image(warp_imgs)
        save_rgbdd(
            output_dir,
            frames_meta["target"],
            np_imgs,
            None, #np_calib_depths,
            np_a_depths,
            des_db,
            warp_imgs,
            save_img=cache_img,
        )
        i += len(frames_meta["target"])
        progress_bar.update(len(frames_meta["target"]))
    json.dump({scene_id: sequence}, open(f"{output_dir}/sequence_{scene_id}.json", "w"))


def check_completed2(out_dir, scene_id):
    json_path = os.path.join(out_dir, scene_id, f"sequence_{scene_id}.json")
    return os.path.exists(json_path)


def check_compeleted(db_path, keys):
    if not os.path.exists(db_path):
        return False
    db = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode() for key in exist_keys]
        depth_keys = [key for key in exist_keys if key.endswith("_depth")]
        img_keys = [key for key in exist_keys if key.endswith("_rgb")]
        pose_keys = [key for key in exist_keys if key.endswith("_pose")]
        remain_poses = keys - set([key.replace("_pose", "") for key in pose_keys])
        remain_imgs = keys - set([key.replace("_rgb", "") for key in img_keys])
        remain_depths = keys - set([key.replace("_depth", "") for key in depth_keys])

    return (
        (len(remain_poses) == 0)
        and (len(remain_imgs) == 0)
        and (len(remain_depths) == 0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--hop_in", type=int, default=4)
    parser.add_argument("--hop_out", type=int, default=4)
    parser.add_argument("--gaussian_clip", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="/work/vig/hieu/gen300/test")
    args = parser.parse_args()

    data_root = "/work/vig/hieu/3dfront_data"
    cfgs = [
        "./configs/base.yaml",
        # "./configs/base_layout_rcn_iodepth.yaml",
        "./configs/base_layoutobj_rcn_iodepth.yaml",
        # "./configs/layout_3dfront_random_pose_iodepth_1871_scene.yaml",
        # "./configs/3dfront_layout_iodepth_1871_scene_3m_final.yaml",
        "./configs/3dfront_layoutobj_iodepth_1871_scene_3m_final_rerender5m.yaml",
        # "./configs/layout_3dfront_random_pose_iodepth.yaml",
    ]
    # ckpt_path = "/work/vig/hieu/escher/logs_eschernet_3dfront_layout_iodepth_1871_scene_3m_obj_final/pipeline-15000"
    ckpt_path = "/work/vig/hieu/escher/logs_eschernet_3dfront_layoutobj_iodepth_1871_scene_novelcontent_finetune/pipeline-4000"

    # ckpt_path = "/work/vig/hieu/escher/logs_eschernet_3dfront_layoutobj_iodepth_1871_scene_3m_final_rerender5m/pipeline-10000"
    # out_dir = "/work/vig/hieu/gen300/iodepth_3m_obj_15k_final_vae_ft"
    # out_dir = "/work/vig/hieu/gen300/iodepth_3m_15k_final_vae_ft_noddim"
    # out_dir = "/work/vig/hieu/gen300/iodepth_3m_obj_15k_final_vae_ft_noddim_custom"
    # out_dir = "/work/vig/hieu/gen300/iodepth_3m_obj_15k_final_vae_ft_noddim_custom2_gauss_clip05"
    out_dir = "/work/vig/hieu/gen300/gen_result_nvfinetune_no_reference_exp"
    os.makedirs(out_dir, exist_ok=True)
    # out_dir = args.out_dir

    # ---------------------------- 0.0 configure the path to necessary data ----------------------------
    configs = [OmegaConf.load(cfg) for cfg in cfgs]
    cfg = OmegaConf.merge(*configs)
    # cfg.data.params.val_scene_ids = f"/work/vig/hieu/escher/6DoF/custom_scenes.json"
    # cfg.data.params.layout_dir = f"{data_root}/layout_pcd_custom_scenes_random_floor"
    # cfg.data.params.val_dir = f"{data_root}/images_custom_scenes_random_floor"
    # cfg.data.params.graph_dir = f"{data_root}/graph_poses_custom"
    # cfg.data.params.pose_dir = f"{data_root}/poses_custom"

    cfg.data.params.val_scene_ids = f"{data_root}/val_scenes_300_3000.json"
    cfg.data.params.layout_dir = f"{data_root}/layout_pcd_3000scenes_random_floor"
    cfg.data.params.val_dir = f"{data_root}/images_3000scenes_random_floor"
    
    # cfg.data.params.datasets_cfg[0].params.val_scene_ids = f"{data_root}/val_scenes_300_3000.json"
    # cfg.data.params.datasets_cfg[0].params.layout_dir = f"{data_root}/layout_pcd_3000scenes_random_floor"
    # cfg.data.params.datasets_cfg[0].params.val_dir = f"{data_root}/images_3000scenes_random_floor"

    # cfg.data.params.val_scene_ids = f"{data_root}/val_scenes_19_100.json"
    # cfg.data.params.layout_dir = f"{data_root}/layout_pcd_100scenes_random_floor"
    # cfg.data.params.val_dir = f"{data_root}/images_100scenes_random_floor"

    cfg.data.params.return_depth_input = False  # ddim inversion
    ddim_inversion = False
    weight_dtype = torch.float16
    image_size = 256

    # ---------------------------- 1.0 configure the pipeline, loading the pre-trained models -----------------------------------------
    # the inference ppl simply takes in the processed input and output the processed output
    device = "cuda"
    pipeline = make_pipeline(
        cfg,
        ckpt_path,
        weight_dtype,
        device,
        inverse_ddim=ddim_inversion,
        vae_ft="/work/vig/hieu/escher/vae-ft-mse-840000-ema-pruned.ckpt",
    )
    depth_model = torch.hub.load(
        "lpiccinelli-eth/UniDepth",
        "UniDepth",
        version="v1",
        backbone="ViTL14",
        pretrained=True,
        trust_repo=True,
        # force_reload=True,
    ).to(device)
    torch_renderer = Torch3DRenderer(image_size, device, radius=3.0)

    val_scene_ids = sorted(json.load(open(cfg.data.params.datasets_cfg[0].params.val_scene_ids, "r")))
    # val_scene_ids = sorted(json.load(open(cfg.data.params.val_scene_ids, "r")))
    # scene_id_file = "/work/vig/hieu/escher/eval_scenes201.json"
    # val_scene_ids = sorted(json.load(open(scene_id_file, "r")))
    val_scene_ids = val_scene_ids[args.start : args.end]
    val_scene_ids = val_scene_ids[args.offset :: args.step]

    val_scene_ids = [
        '64cce374-230b-4fe2-8240-69f81c8cfb33',
        '6634054e-6ff5-43d2-958f-a80bb7eee357', 
        '6b56575f-a746-4062-8296-b566d5de7b60', 
        '651c37ce-c0cd-47c6-843f-3aa192235e39'
        ]
    for scene_id in tqdm(val_scene_ids):
        if check_completed2(out_dir, scene_id):
            print(f"scene {scene_id} is already completed")
            continue
        main_ddim_inversion(
            cfg,
            os.path.join(out_dir, scene_id),
            scene_id,
            cond_hop=args.hop_in,
            target_hop=args.hop_out,
            device=device,
            pipeline=pipeline,
            depth_model=depth_model,
            torch_renderer=torch_renderer,
            max_view=60,
            cache_img=False,  ######################
            ddim_inversion=ddim_inversion,
        )

"""
CUDA_VISIBLE_DEVICES=0 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 0
CUDA_VISIBLE_DEVICES=1 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 1
CUDA_VISIBLE_DEVICES=2 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 2
CUDA_VISIBLE_DEVICES=3 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 3

CUDA_VISIBLE_DEVICES=4 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 4
CUDA_VISIBLE_DEVICES=5 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 5
CUDA_VISIBLE_DEVICES=6 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 6
CUDA_VISIBLE_DEVICES=7 python generate_scene_absolute_depth_no_cond2.py --step 8 --offset 7

CUDA_VISIBLE_DEVICES=7 python generate_scene_absolute_depth_no_cond2.py --step 8 --hop_out 1 --out_dir /work/vig/hieu/gen300/iodepth_3m_15k_final_vae_ft_4in_1out --offset 7

4in 3out
CUDA_VISIBLE_DEVICES=7 python generate_scene_absolute_depth_no_cond2.py --step 8 --hop_out 3 --out_dir /work/vig/hieu/gen300/iodepth_3m_15k_final_vae_ft_4in_3out --offset 7

finishing 3in 4 out done
CUDA_VISIBLE_DEVICES=0 python generate_scene_absolute_depth_no_cond2.py --hop_in 3 --out_dir /work/vig/hieu/gen300/iodepth_3m_15k_final_vae_ft_3in_4out --offset 0

finishing 4in 2 out
CUDA_VISIBLE_DEVICES=7 python generate_scene_absolute_depth_no_cond2.py --step 64 --hop_out 2 --out_dir /work/vig/hieu/gen300/iodepth_3m_15k_final_vae_ft_4in_2out --offset 56


1in 4out
CUDA_VISIBLE_DEVICES=0 python generate_scene_absolute_depth_no_cond2.py --step 1 --hop_in 3 --out_dir /work/vig/hieu/gen300/iodepth_3m_15k_final_vae_ft_3in_4out --offset 0
"""
