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
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
            a_depth_path = f"{out_dir}/absolute_depth/{frame_id}.png"
            plt.imsave(img_path, img)
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


def make_dataset(cfg, out_dir_db, scene_id, data_root):
    data_kw = OmegaConf.to_container(cfg.data.params)

    data_kw["target_dir"] = data_kw["val_dir"]

    for key in ["train_dir", "val_dir", "dataset_cls", "batch_size", "num_workers"]:
        if key in data_kw:
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
    data_kw["inference_ppl"] = True
    data_kw["data_root"] = data_root

    # scene_id = val_scene_ids[0] #"031fba32-7c48-4c1e-8342-aab66d6e531f"
    data_kw["scene_ids"] = [scene_id]
    print('making dataset for scene:', data_kw["scene_ids"])
    out_db_dir = os.path.join(out_dir_db, scene_id)
    os.makedirs(out_db_dir, exist_ok=True)
    dataset = Front3DPose(**data_kw)
    return dataset

def make_dataset2(cfg, out_dir_db, scene_id, data_root):
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
        if key in data_kw:
            data_kw.pop(key)
        if key in data_kw2:
            data_kw2.pop(key)
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
    data_kw["data_root"] = data_root

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


def setup_out_db(dataset, scene_id, anchors=None):
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
        exist_keys_rgb = [key for key in exist_keys if key.endswith("_rgb")]
        exist_keys = [key for key in exist_keys if key.endswith("_pose")]
        # print(len(exist_keys))
        for key in exist_keys:
            # key = key.replace("_rgb", "_pose")
            rgb_key = key.replace("_pose", "_rgb")
            if not rgb_key in exist_keys_rgb:
                continue
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
    print(f'graph has {len(in_key)} keys')
    # print(f'graph has {len(graph.nodes)} nodes')
    furniture_graph = graph.copy()
    cc = sorted(
        [c for c in nx.connected_components(graph)], key=lambda x: len(x), reverse=True
    )
    print(f'largest connected component has {len(cc[0])} nodes')
    graph = graph.subgraph(cc[0]).copy()
    # print(f'the sampled subgraph has {len(graph.nodes)} nodes')
    # print('the nodes are:', sorted(cc[0])[0:10], "...")

    graph_with_wall = dataset.complete_graphs[scene_id].subgraph(in_key)
    cc_with_wall = sorted(
        [c for c in nx.connected_components(graph_with_wall)],
        key=lambda x: len(x),
        reverse=True,
    )
    graph_with_wall = graph_with_wall.copy()
    graph_with_wall_connected = graph_with_wall.subgraph(cc_with_wall[0]).copy()
    print(f'largest connected component has {len(cc_with_wall[0])} nodes')

    # cond view for bedroom
    # "031fba32-7c48-4c1e-8342-aab66d6e531f"
    # cond_view = sorted(in_key)[0]

    # cond view for living room
    # "031fba32-7c48-4c1e-8342-aab66d6e531f"
    # cond_view = sorted(in_key)[-5]
    
    if anchors is not None:
        cond_views = anchors
        for view in cond_views:
            make_dummy_view(src_db, des_db, view)
            cond_view = sorted(cc[0])[0]
    else:
        cond_view = sorted(cc[0])[0]
        make_dummy_view(src_db, des_db, cond_view)
    copy_pose(src_db, des_db, in_key)
    return graph, cond_view, des_db, graph_with_wall, furniture_graph


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

    # -----0.0 set up the dataset, load the graph, and set up the output db-----
    self_defiend_anchors = None
    # anchoring_views = {
    #     "64cce374-230b-4fe2-8240-69f81c8cfb33": ['0699','0868','1146'],
    # }
    # if scene_id in anchoring_views:
    #     self_defiend_anchors = anchoring_views[scene_id]
    data_root = cfg.get('data_root', "/projects/vig/Datasets/3D-Front/render_data")
    dataset = make_dataset(cfg, os.path.join(output_dir, "db"), scene_id, data_root)
    # load the graphs, graph will be the nodes that only contain the target view with furnitures, 
    # graph_with_wall will be the nodes that contain the target view with furnitures and walls
    graph, cond_view, des_db, graph_with_wall, furniture_graph = setup_out_db(dataset, scene_id, anchors = self_defiend_anchors)
    keys = set(list(graph.nodes))
    if check_compeleted(os.path.join(output_dir, "db", scene_id), keys):
        print(f"scene {scene_id} is already completed")
        return
    depth_func = lambda x: depth_model.infer(x[None, ...])["depth"][0, 0]
    depth_transform = instantiate_from_config(cfg.data.params.depth_tform_cfg)
    i = 0
    # special choice of condition and target for the first node
    # NOTE either chose graph search 1 or 2
    # grpah_search: util class for manaing the graph
    graph_search = GraphSearch2(
        furniture_graph, cond_view, cond_hop=cond_hop, target_hop=target_hop
    )
    max_depth = cfg.data.params.depth_tform_cfg.params.max_depth
    progress_bar = tqdm(total=len(furniture_graph.nodes))
    sequence = []

    # -----1.0 loop through the wall_only graph-----
    # todo: change the graph sampling strategy so we can as many target views as possible
    while True:
        torch.cuda.empty_cache()
        # target node is chosen to be a node in the candidate set
        if self_defiend_anchors is not None and len(self_defiend_anchors) > 0:
            target_node = self_defiend_anchors.pop(0)

            generating_anchor = True
            cond_view = target_node
        else:
            target_node = graph_search.next_node()
            generating_anchor = False
        if target_node is None:
            break
        # sample the condition and target view in the target node's neighborhood
        cond, target = graph_search.get_cond_target(target_node)
        if i == 0 and not generating_anchor:

            assert len(cond) == 0
            assert cond_view in target
            cond = [cond_view]
            target = [t for t in target if t != cond_view]
        if generating_anchor:
            assert cond_view in target
            if cond is None:
                cond = [cond_view]
            else:
                cond.extend([cond_view])
            target = [t for t in target if t != cond_view]
        if len(cond) < 1:
            cond = target[:1]
            target = target[1:]
        target = target[:max_view]
        for n in target:
            graph_search.set_visited(n)

        frames_meta = {
            "cond": cond,
            "target": target,
        }
        # print('current cond:', len(cond), cond[:5],"...")
        # print('current target:', len(target),target[:5],"...")
        batch = dataset.get_item_from_meta(scene_id, frames_meta, anchor = True)
        # print('calling get_layout from the dataset')
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

    # -----2.0 loop through the frames with walls-----
    graph_search_with_wall = GraphSearch2(
        graph_with_wall, cond_view, cond_hop=cond_hop, target_hop=target_hop
    )
    progress_bar_sweep = tqdm(total=len(graph_with_wall.nodes))
    # set all previously visited nodes in graph to be visited
    for n in furniture_graph.nodes:
        if n in graph_search_with_wall.graph.nodes:
            graph_search_with_wall.set_visited(n)

    # todo: delete this if training on a6000:
    while True:
        target_node = graph_search_with_wall.next_node()
        if target_node is None:
            break
        cond, target = graph_search_with_wall.get_cond_target(target_node)
        # cond = cond[:max_view]
        cond_with_wall = [c for c in cond if c in graph.nodes]
        if len(cond_with_wall) < max_view:
            cond = cond_with_wall + cond[: max_view - len(cond_with_wall)]
        else:
            cond = cond_with_wall[:max_view]
        target = target[:max_view]
        for n in target:
            graph_search_with_wall.set_visited(n)

        if len(cond) < 1:
            cond = target[:1]
            if len(target) > 1:
                target = target[1:]
            else:
                target = cond
            anchor = True
        else:
            anchor = False

        frames_meta = {
            "cond": cond,
            "target": target,
        }
        batch = dataset.get_item_from_meta(scene_id, frames_meta, anchor=anchor)
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
        progress_bar_sweep.update(len(frames_meta["target"]))
                
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
    parser.add_argument("--hop_in", type=int, default=8)
    parser.add_argument("--hop_out", type=int, default=8)
    parser.add_argument("--gaussian_clip", type=float, default=0.0)
    parser.add_argument("--data_root", type=str, default="../dataRelease")
    parser.add_argument("--ckpt_path", type=str, default="../ckpts/3dfront_layout_iodepth_1871_scene_3m")
    parser.add_argument("--out_dir", type=str, default="../gen_rgbd")
    args = parser.parse_args()

    data_root = args.data_root
    cfgs = [
        "./configs/base.yaml",
        # "./configs/base_layout_rcn_iodepth.yaml",
        "./new_explorer_configs/base_layout_rcn_iodepth_v.yaml",        
        # "./configs/layout_3dfront_random_pose_iodepth_1871_scene.yaml",
        # "./configs/3dfront_layout_iodepth_1871_scene_3m_final.yaml",
        "./new_explorer_configs/3dfront_layout_rand_curate_explorer.yaml",
        # "./configs/layout_3dfront_random_pose_iodepth.yaml",
    ]
    ckpt_path = args.ckpt_path
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------- 0.0 configure the path to necessary data ----------------------------
    configs = [OmegaConf.load(cfg) for cfg in cfgs]
    cfg = OmegaConf.merge(*configs)

    # Store data_root in cfg for later use
    cfg.data_root = data_root
    
    cfg.data.params.val_scene_ids = f"{data_root}/val_scenes_300_3000.json"
    cfg.data.params.layout_dir = f"{data_root}/layout_samples"
    cfg.data.params.val_dir = f"{data_root}/rendered_floor_sample"
    
    cfg.data.params.return_depth_input = False  # ddim inversion
    cfg.data.params.graph_dir = f"{data_root}/graph_poses_all"
    cfg.data.params.curation_meta = f"{data_root}/wall_info_all"

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
        vae_ft="../ckpts/vae-ft-mse-840000-ema-pruned.ckpt",
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
    val_scene_ids = val_scene_ids[args.start : args.end]
    val_scene_ids = val_scene_ids[args.offset :: args.step]

    val_scene_ids = [
        '64cce374-230b-4fe2-8240-69f81c8cfb33',
        '644f3b6e-ad35-4254-92ea-626e3e8a65b1', 
        '65aa5c84-5d84-4785-adf8-0000c91aa79e',
        '64cce374-230b-4fe2-8240-69f81c8cfb33',
        '651c37ce-c0cd-47c6-843f-3aa192235e39',
        '644dba87-0d65-4897-912c-38185791b3c2',
        '6a03badb-61ab-4e22-9564-de2fd2ed1938',
        '6b56575f-a746-4062-8296-b566d5de7b60',
        '64b7a725-7b94-443b-a8b1-d582ba1cd3d9',
        '6ae5c274-8d8c-4832-878c-ea4f74082dfa',
        '64e2e0ba-3769-479d-81d7-c870da620b07',
        '6634054e-6ff5-43d2-958f-a80bb7eee357', 
        '6b5bb08d-7e14-4206-8014-685580c674b1',
        '6bb9d1bf-8fd5-4278-b4a0-38ea346dafcd',
        '6be7ef9a-98d9-4d32-8b0c-c0d2b1ca3bbe',
        '6c2d1fd6-0cb5-4382-9788-c0d95431a6ec',
        '64ac0b68-fc6c-490d-ab08-6f45e947b4a7',
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

