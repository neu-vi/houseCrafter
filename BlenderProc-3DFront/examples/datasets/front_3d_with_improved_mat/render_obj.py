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
    parser.add_argument("--scene_id", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--valid_pose_folder", type=str, default="")
    parser.add_argument("--poses_per_obj", type=int, default=20)
    parser.add_argument("--visible_thr", type=int, default=1000)
    parser.add_argument("--error_folder", type=str, default="./error")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--depth", type=str, default="1")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()


def save_data_per_pose_batch(db, data, poses, keys):
    """
    db: lmdb environment
    data: render results for poses at a location
    """
    out_txn = db.begin(write=True)
    depths = data.get("depth", [None]*len(keys))
    for key, pose, rgb, depth in zip(keys, poses, data["colors"], depths):
        rgb_key = f"{key}_rgb".encode("ascii")
        rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        out_txn.put(rgb_key, rgb)

        pose_key = f"{key}_pose".encode("ascii")
        pose = zlib.compress(pose.astype(np.float32).tobytes())
        out_txn.put(pose_key, pose)
        if depth is not None:
            depth_key = f"{key}_depth".encode("ascii")
            depth[depth > MAX_DEPTH] = 0.0
            # save depth in mm as uint16
            depth = (depth * 1000).astype(np.uint16)
            depth = zlib.compress(depth.tobytes())
            out_txn.put(depth_key, depth)
    out_txn.commit()


def setup_blender(args):
    bproc.init()
    RendererUtility.set_max_amount_of_samples(32)
    if args.depth == "1":
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
    RendererUtility.set_cpu_threads(args.cpu_threads)
    # set the light bounces
    bproc.renderer.set_light_bounces(
        diffuse_bounces=200,
        glossy_bounces=200,
        max_bounces=200,
        transmission_bounces=200,
        transparent_max_bounces=200,
    )
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
    cc_materials = bproc.loader.load_ccmaterials(
        args.cc_material_folder, ["Bricks", "Wood", "Carpet", "Tile", "Marble"]
    )

    # Sample materials for floors
    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    floors = sorted(floors, key=lambda x: x.get_name())
    rg = np.random.RandomState(seed)
    for floor in floors:
        for i in range(len(floor.get_materials())):
            floor.set_material(i, rg.choice(cc_materials))

    # Sample materials for baseboards_and_doors
    baseboards_and_doors = bproc.filter.by_attr(
        loaded_objects, "name", "Baseboard.*|Door.*", regex=True
    )
    baseboards_and_doors = sorted(baseboards_and_doors, key=lambda x: x.get_name())
    wood_floor_materials = bproc.filter.by_cp(
        cc_materials, "asset_name", "WoodFloor.*", regex=True
    )
    wood_floor_materials = sorted(
        wood_floor_materials, key=lambda x: x.get_cp("asset_name")
    )
    for obj in baseboards_and_doors:
        for i in range(len(obj.get_materials())):
            obj.set_material(i, rg.choice(wood_floor_materials))

    # Sample materials for walls
    walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
    walls = sorted(walls, key=lambda x: x.get_name())
    marble_materials = bproc.filter.by_cp(
        cc_materials, "asset_name", "Marble.*", regex=True
    )
    marble_materials = sorted(marble_materials, key=lambda x: x.get_cp("asset_name"))
    for wall in walls:
        for i in range(len(wall.get_materials())):
            wall.set_material(i, rg.choice(marble_materials))

    return loaded_objects


def render_scene_from_pose(scene_id, args, seed):
    setup_blender(args)
    poses_by_obj = pickle.load(
        open(os.path.join(args.poses_folder, f"{scene_id}.pkl"), "rb")
    )  # instance_id -> poses
    poses_dict = {}
    for instance_id, poses in poses_by_obj.items():
        instance_id = instance_id.replace("/", "-")
        poses_dict[instance_id] = {
            f"{instance_id}--{i:0>4}": loc for i, loc in enumerate(poses["poses"])
        }

    # TODO get invalid pose
    scene_objects = None
    invalid_poses_path = os.path.join(args.poses_folder, f"{scene_id}_invalid.json")
    if os.path.exists(invalid_poses_path):
        with open(invalid_poses_path, "r") as f:
            invalid_poses = set(json.load(f))
    else:
        scene_objects = setup_data(args, scene_id, seed)
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(
            [o for o in scene_objects if isinstance(o, bproc.types.MeshObject)]
        )
        invalid_poses = set()
        for poses in poses_dict.values():
            for key, pose in tqdm(poses.items()):
                if not bproc.camera.perform_obstacle_in_view_check(
                    pose, {"min": 0.1}, bvh_tree
                ):
                    invalid_poses.add(key)
        json.dump(list(invalid_poses), open(invalid_poses_path, "w"))

    # check finished objects
    valid_pose_folder = os.path.join(args.valid_pose_folder, scene_id)
    os.makedirs(valid_pose_folder, exist_ok=True)
    finished_objects = os.listdir(valid_pose_folder)
    finished_objects = set([obj.split(".")[0] for obj in finished_objects])
    for obj in finished_objects:
        poses_dict.pop(obj, None)

    # filter invalid locations, skip computed locations
    for obj_id, poses in poses_dict.items():
        keys = list(poses.keys())
        for key in keys:
            if key in invalid_poses:
                poses.pop(key, None)
    obj_ids = list(poses_dict.keys())
    for obj_id in obj_ids:
        if not len(poses_dict[obj_id]):
            poses_dict.pop(obj_id, None)

    if not len(poses_dict):
        print("scene {scene_id} completed.")
        return
    else:
        print(f"scene {scene_id} has {len(poses)} poses left to render.")

    if scene_objects is None:
        scene_objects = setup_data(args, scene_id, seed)

    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin(write=False) as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode() for key in exist_keys]
        exist_keys = [key[: key.rfind("_")] for key in exist_keys]
        key_counters = Counter(exist_keys)
        if args.depth == "1":
            exist_keys = set([key for key, count in key_counters.items() if count == 3])
        else:
            exist_keys = set([key for key, count in key_counters.items() if count >= 2])

    for obj_id, poses in poses_dict.items():
        valid_poses = []
        for key, pose in poses.items():
            if key in exist_keys:
                valid_poses.append(key)
                if len(valid_poses) == args.poses_per_obj:
                    break
                continue
            bproc.utility.reset_keyframes()
            bproc.camera.add_camera_pose(pose)

            # check valid pose by render semseg
            default_values = {"cp_instanceid": ""}
            seg_data = bproc.renderer.render_segmap(
                map_by=["cp_instanceid", "instance"], default_values=default_values
            )
            mask_size = 0
            for instance in seg_data["instance_attribute_maps"][0]:
                if instance["instanceid"] == obj_id:
                    mask = seg_data["instance_segmaps"][0] == instance["idx"]
                    mask_size += mask.sum()
            if mask_size < args.visible_thr:
                continue

            # render rgbd
            data = bproc.renderer.render()
            save_data_per_pose_batch(env, data, [pose], [key])

            valid_poses.append(key)
            if len(valid_poses) == args.poses_per_obj:
                break
        json.dump(valid_poses, open(f"{valid_pose_folder}/{obj_id}.json", "w"))
    env.close()


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
    # if scene_id in error_scenes:
    #     exit(0)
    try:
    # if True:
        # if args.locations_folder:
        #     print("run location")
        #     render_scene_from_location(scene_id, args, seed=args.seed)
        if args.poses_folder:
            render_scene_from_pose(scene_id, args, seed=args.seed)
    except Exception as e:
        with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
            f.write(str(e))
        print(f"Error in scene {scene_id}")
