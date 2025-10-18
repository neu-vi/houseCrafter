import blenderproc as bproc
import blenderproc.python.renderer.RendererUtility as RendererUtility

import argparse
import json
import math
import os
import zlib
from collections import Counter
import pickle
import cv2
import lmdb
import numpy as np
import quaternion as qt
from tqdm import tqdm
from glob import glob

# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

MAX_DEPTH = 60.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument(
        "front_3D_texture_folder", help="Path to the 3D FRONT texture folder."
    )
    parser.add_argument(
        "cc_material_folder",
        nargs="?",
        default="resources/cctextures",
        help="Path to CCTextures folder, see the /scripts for the download script.",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default="examples/datasets/front_3d_with_improved_mat/renderings",
        help="Path to where the data should be saved",
    )
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--cpu_threads", type=int, required=True)
    parser.add_argument("--scene_id", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--valid_pose_folder", type=str, default="")
    parser.add_argument("--poses_per_obj", type=int, default=20)
    parser.add_argument("--visible_thr", type=int, default=1000)
    parser.add_argument("--error_folder", type=str, default="./error")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()





def setup_blender(args):
    bproc.init()
    RendererUtility.set_noise_threshold(0)
    RendererUtility.set_denoiser(None)
    RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
    RendererUtility.set_max_amount_of_samples(1)
    RendererUtility.set_cpu_threads(args.cpu_threads)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # set intrinsic parameters
    bproc.camera.set_intrinsics_from_blender_params(
        lens=np.radians(args.fov),
        image_width=args.res_x,
        image_height=args.res_y,
        lens_unit="FOV",
    )

    cam_K = bproc.camera.get_intrinsics_as_K_matrix()
    cam_intrinsic_path = os.path.join(args.output_folder, "cam_K.npy")
    # write camera intrinsics
    if not os.path.exists(cam_intrinsic_path):
        np.save(str(cam_intrinsic_path), cam_K)


def setup_data(args, scene_id, seed):

    mapping_file = bproc.utility.resolve_resource(
        os.path.join("front_3D", "blender_label_mapping.csv")
    )
    label_mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # read 3d future model info
    with open(os.path.join(args.future_folder, "model_info_revised.json"), "r") as f:
        model_info_data = json.load(f)
    model_id_to_label = {
        m["model_id"]: (
            m["category"].lower().replace(" / ", "/") if m["category"] else "others"
        )
        for m in model_info_data
    }

    # load the front 3D objects
    front_json = os.path.join(args.front_folder, f"{scene_id}.json")
    loaded_objects = bproc.loader.load_front3d2(
        json_path=front_json,
        future_model_path=args.future_folder,
        front_3D_texture_path=args.front_3D_texture_folder,
        label_mapping=label_mapping,
        model_id_to_label=model_id_to_label,
    )

    # -------------------------------------------------------------------------
    #          Sample materials
    # -------------------------------------------------------------------------
    # cc_materials = bproc.loader.load_ccmaterials(
    #     args.cc_material_folder, ["Bricks", "Wood", "Carpet", "Tile", "Marble"]
    # )

    # # Sample materials for floors
    # floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    # floors = sorted(floors, key=lambda x: x.get_name())
    # rg = np.random.RandomState(seed)
    # for floor in floors:
    #     for i in range(len(floor.get_materials())):
    #         floor.set_material(i, rg.choice(cc_materials))

    # # Sample materials for baseboards_and_doors
    # baseboards_and_doors = bproc.filter.by_attr(
    #     loaded_objects, "name", "Baseboard.*|Door.*", regex=True
    # )
    # baseboards_and_doors = sorted(baseboards_and_doors, key=lambda x: x.get_name())
    # wood_floor_materials = bproc.filter.by_cp(
    #     cc_materials, "asset_name", "WoodFloor.*", regex=True
    # )
    # wood_floor_materials = sorted(
    #     wood_floor_materials, key=lambda x: x.get_cp("asset_name")
    # )
    # for obj in baseboards_and_doors:
    #     for i in range(len(obj.get_materials())):
    #         obj.set_material(i, rg.choice(wood_floor_materials))

    # # Sample materials for walls
    # walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
    # walls = sorted(walls, key=lambda x: x.get_name())
    # marble_materials = bproc.filter.by_cp(
    #     cc_materials, "asset_name", "Marble.*", regex=True
    # )
    # marble_materials = sorted(marble_materials, key=lambda x: x.get_cp("asset_name"))
    # for wall in walls:
    #     for i in range(len(wall.get_materials())):
    #         wall.set_material(i, rg.choice(marble_materials))

    return loaded_objects


def render_scene_from_pose(scene_id, args, seed, batch_size=12):
    setup_blender(args)
    valid_pose_files = list(glob(f"{args.valid_pose_folder}/{scene_id}/*.json"))
    valid_poses = sum([json.load(open(f)) for f in valid_pose_files], [])
    valid_poses = set(valid_poses)

    poses_by_obj = pickle.load(
        open(os.path.join(args.poses_folder, f"{scene_id}.pkl"), "rb")
    )  # instance_id -> poses
    poses_dict = {}
    for instance_id, poses in poses_by_obj.items():
        instance_id = instance_id.replace("/", "-")
        poses_dict.update(
            {f"{instance_id}--{i:0>4}": loc for i, loc in enumerate(poses["poses"])}
        )
    poses = {k: v for k, v in poses_dict.items() if k in valid_poses}

    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode() for key in exist_keys]
        exist_keys = [key for key in exist_keys if not key.endswith("_pose")]
        exist_keys = [key.split("_")[0] for key in exist_keys]
        exist_keys = set(exist_keys)

    for key in exist_keys:
        poses.pop(key, None)

    if not len(poses):
        print("scene {scene_id} completed.")
        return
    else:
        print(f"scene {scene_id} has {len(poses)} poses left to render.")
        
    setup_data(args, scene_id, seed)
        
        
    keys = list(poses.keys())
    n_batches = int(math.ceil(len(keys) / batch_size))
    for i in range(n_batches):
        batch_keys = keys[i * batch_size : (i + 1) * batch_size]
        batch_poses = [poses[key] for key in batch_keys]
        bproc.utility.reset_keyframes()
        for pose in batch_poses:
            bproc.camera.add_camera_pose(pose)
        data = bproc.renderer.render()
        save_data_per_pose_batch(env, data, batch_poses, batch_keys)


def save_data_per_pose_batch(db, data, poses, keys):
    """
    db: lmdb environment
    data: render results for poses at a location
    """
    # TODO: remove pose and rgb later
    out_txn = db.begin(write=True)
    for key, pose, rgb, depth in zip(keys, poses, data["colors"], data["depth"]):
        # rgb_key = f"{key}_rgb".encode("ascii")
        # rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        # out_txn.put(rgb_key, rgb)

        pose_key = f"{key}_pose".encode("ascii")
        pose = zlib.compress(pose.astype(np.float32).tobytes())
        out_txn.put(pose_key, pose)

        depth_key = f"{key}_depth".encode("ascii")
        depth[depth > MAX_DEPTH] = 0.0
        # save depth in mm as uint16
        depth = (depth * 1000).astype(np.uint16)
        depth = zlib.compress(depth.tobytes())
        out_txn.put(depth_key, depth)
    out_txn.commit()


if __name__ == "__main__":
    args = parse_args()
    assert args.locations_folder or args.poses_folder, "Need locations or poses folder"
    assert not (
        args.locations_folder and args.poses_folder
    ), "Only one of locations or poses folder should be provided"
    os.makedirs(args.output_folder, exist_ok=True)
    scene_id = args.scene_id
    os.makedirs(args.error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.error_folder)]
    if scene_id in error_scenes:
        exit(0)
    try:
        # if args.locations_folder:
        #     print("run location")
        #     render_scene_from_location(scene_id, args, seed=args.seed)
        if args.poses_folder:
            render_scene_from_pose(scene_id, args, seed=args.seed)
    except Exception as e:
        with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
            f.write(str(e))
        print(f"Error in scene {scene_id}")
    # scene_id = "6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9"
    # if args.locations_folder:
    #     render_scene_from_location(scene_id, args, seed=0)
    # if args.poses_folder:
    #     render_scene_from_pose(scene_id, args, seed=0)

