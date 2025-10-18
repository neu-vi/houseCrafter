import blenderproc as bproc
import blenderproc.python.renderer.RendererUtility as RendererUtility

print()

import argparse
import json
import math
import os
import pickle
import zlib
from collections import Counter

import cv2
import lmdb
import numpy as np
import quaternion as qt
from einops import repeat
from tqdm import tqdm

print()
# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()

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

    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--locations_graph_folder", type=str, default="")
    parser.add_argument("--error_folder", type=str, default="./error")

    parser.add_argument("--n_heading", type=int, default=5)
    parser.add_argument("--elevation", type=float, default=40)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--depth", type=str, default="0")
    parser.add_argument("--scene_id", type=str)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()


def get_camera_poses(location: np.ndarray) -> np.ndarray:
    """
    select poses from the given location

    try 2 settings
        12 views, 0 deg elevation
        12 views, 6 look down 30 deg 6 look up 30 deg

    NOTE: assume y is up
    # cam coor in blender is x right, y up, z backward
    """
    HEADINGS = np.arange(12) * 2 * np.pi / 12
    # rotation for setting 1
    ROTATIONS1 = qt.from_rotation_vector(HEADINGS[:, np.newaxis] * np.array([0, 1, 0]))
    ROTATIONS1 = qt.as_rotation_matrix(ROTATIONS1)

    poses = np.zeros((len(HEADINGS), 4, 4), dtype=np.float32)
    poses[:, :3, :3] = ROTATIONS1
    poses[:, :3, 3] += location
    poses[:, 3, 3] = 1
    return poses


def get_camera_poses2(location, n_heading=5, elevation=40):
    HEADINGS = np.arange(n_heading) * 2 * np.pi / n_heading
    ELEVATION = np.radians(elevation)
    ELEVATIONS = np.array([ELEVATION, -ELEVATION])
    HEADINGS = repeat(HEADINGS, "n -> (e n)", e=2)
    ELEVATIONS = repeat(ELEVATIONS, "e -> (e n)", n=n_heading)
    # rotation for setting 1
    ROTATIONS1 = qt.from_rotation_vector(HEADINGS[:, np.newaxis] * np.array([0, 1, 0]))

    ROTATIONS2 = qt.from_rotation_vector(
        ELEVATIONS[:, np.newaxis] * np.array([1, 0, 0])
    )
    R = qt.as_rotation_matrix(ROTATIONS1 * ROTATIONS2)

    poses = np.zeros((len(HEADINGS), 4, 4), dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:, :3, 3] += location
    poses[:, 3, 3] = 1
    return poses


def check_name(name, category_name):
    return True if category_name in name.lower() else False


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


def render_scene_from_location(scene_id, args, seed):
    POSES_PER_LOCATION = 12
    setup_blender(args)
    locations = np.load(os.path.join(args.locations_folder, f"{scene_id}.npy"))
    locations = {f"{i:0>4}": loc for i, loc in enumerate(locations)}
    # TODO get invalid location
    scene_objects = None
    invalid_locations_path = os.path.join(
        args.locations_folder, f"{scene_id}_invalid.json"
    )
    if os.path.exists(invalid_locations_path):
        with open(invalid_locations_path, "r") as f:
            invalid_locations = set(json.load(f))
    else:
        scene_objects = setup_data(args, scene_id, seed)
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(
            [o for o in scene_objects if isinstance(o, bproc.types.MeshObject)]
        )
        invalid_locations = set()
        for key, loc in tqdm(locations.items()):
            valid = True
            poses = get_camera_poses(loc)
            for pose in poses:
                if not bproc.camera.perform_obstacle_in_view_check(
                    pose, {"min": 0.1}, bvh_tree
                ):
                    valid = False
                    break
            if not valid:
                invalid_locations.add(key)
        json.dump(list(invalid_locations), open(invalid_locations_path, "w"))

    # TODO: connect to db, get computed locations
    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode().split("_")[0] for key in exist_keys]
        key_counters = Counter(exist_keys)
        exist_keys = [
            key
            for key, count in key_counters.items()
            if count == 3 * POSES_PER_LOCATION
        ]

    # filter invalid locations, skip computed locations
    for key in invalid_locations:
        locations.pop(key, None)
    for key in exist_keys:
        locations.pop(key, None)

    if not len(locations):
        print("scene {scene_id} completed.")
        return
    else:
        print(f"scene {scene_id} has {len(locations)} locations left to render.")

    if scene_objects is None:
        scene_objects = setup_data(args, scene_id, seed)

    for key, loc in tqdm(locations.items()):
        bproc.utility.reset_keyframes()

        # set camera pose
        poses = get_camera_poses(loc)
        for pose in poses:
            bproc.camera.add_camera_pose(pose)
        data = bproc.renderer.render()
        save_data_per_location(env, data, poses, key)


def render_scene_from_location_graph(scene_id, args, seed):
    POSES_PER_LOCATION = args.n_heading * 2
    setup_blender(args)
    location_graph = pickle.load(
        open(os.path.join(args.locations_graph_folder, f"{scene_id}.pkl"), "rb")
    )
    keys = sorted(list(location_graph.nodes))
    locations = {key: np.array(location_graph.nodes[key]["pos"]) for key in keys}

    # scene_objects = None
    # invalid_locations_path = os.path.join(
    #     args.locations_folder, f"{scene_id}_invalid.json"
    # )
    # if os.path.exists(invalid_locations_path):
    #     with open(invalid_locations_path, "r") as f:
    #         invalid_locations = set(json.load(f))
    # else:
    #     scene_objects = setup_data(args, scene_id, seed)
    #     bvh_tree = bproc.object.create_bvh_tree_multi_objects(
    #         [o for o in scene_objects if isinstance(o, bproc.types.MeshObject)]
    #     )
    #     invalid_locations = set()
    #     for key, loc in tqdm(locations.items()):
    #         valid = True
    #         poses = get_camera_poses(loc)
    #         for pose in poses:
    #             if not bproc.camera.perform_obstacle_in_view_check(
    #                 pose, {"min": 0.1}, bvh_tree
    #             ):
    #                 valid = False
    #                 break
    #         if not valid:
    #             invalid_locations.add(key)
    #     json.dump(list(invalid_locations), open(invalid_locations_path, "w"))

    # TODO: connect to db, get computed locations
    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode().split("-")[0] for key in exist_keys]
        key_counters = Counter(exist_keys)
        if args.depth == "1":
            exist_keys = [
                key
                for key, count in key_counters.items()
                if count == 3 * POSES_PER_LOCATION
            ]
        else:
            exist_keys = [
                key
                for key, count in key_counters.items()
                if count == 2 * POSES_PER_LOCATION
            ]
    # filter invalid locations, skip computed locations
    # for key in invalid_locations:
    #     locations.pop(key, None)
    for key in exist_keys:
        locations.pop(key, None)

    if not len(locations):
        print("scene {scene_id} completed.")
        return
    else:
        print(f"scene {scene_id} has {len(locations)} locations left to render.")

    # if scene_objects is None:
    scene_objects = setup_data(args, scene_id, seed)

    for key, loc in tqdm(locations.items()):
        bproc.utility.reset_keyframes()

        # set camera pose
        poses = get_camera_poses2(loc, args.n_heading, args.elevation)
        for pose in poses:
            bproc.camera.add_camera_pose(pose)
        data = bproc.renderer.render()
        save_data_per_location(env, data, poses, key)


def render_scene_from_pose(scene_id, args, seed, batch_size=12):
    setup_blender(args)
    poses = np.load(os.path.join(args.poses_folder, f"{scene_id}.npy"))
    poses = {f"{i:0>4}": loc for i, loc in enumerate(poses)}

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
        for key, pose in tqdm(poses.items()):
            if not bproc.camera.perform_obstacle_in_view_check(
                pose, {"min": 0.1}, bvh_tree
            ):
                invalid_poses.add(key)
        json.dump(list(invalid_poses), open(invalid_poses_path, "w"))

    # TODO: connect to db, get computed poses
    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode().split("_")[0] for key in exist_keys]
        key_counters = Counter(exist_keys)
        exist_keys = [key for key, count in key_counters.items() if count == 3]

    # filter invalid locations, skip computed locations
    for key in invalid_poses:
        poses.pop(key, None)
    for key in exist_keys:
        poses.pop(key, None)

    if not len(poses):
        print("scene {scene_id} completed.")
        return
    else:
        print(f"scene {scene_id} has {len(poses)} poses left to render.")

    if scene_objects is None:
        scene_objects = setup_data(args, scene_id, seed)

    keys = list(poses.keys())
    n_batches = int(math.ceil(len(keys) / batch_size))
    for i in range(n_batches):
        batch_keys = keys[i * batch_size : (i + 1) * batch_size]
        bproc.utility.reset_keyframes()
        # set camera pose
        batch_poses = [poses[key] for key in batch_keys]
        for pose in batch_poses:
            bproc.camera.add_camera_pose(pose)
        data = bproc.renderer.render()
        save_data_per_pose_batch(env, data, batch_poses, batch_keys)


def save_data_per_location(db, data, poses, loc_key):
    """
    db: lmdb environment
    data: render results for poses at a location
    """
    out_txn = db.begin(write=True)
    depths = data.get("depth", [None] * len(poses))
    for i, (pose, rgb, depth) in enumerate(zip(poses, data["colors"], depths)):
        rgb_key = f"{loc_key}-{i:0>2}_rgb".encode("ascii")
        rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        out_txn.put(rgb_key, rgb)

        pose_key = f"{loc_key}-{i:0>2}_pose".encode("ascii")
        pose = zlib.compress(pose.astype(np.float32).tobytes())
        out_txn.put(pose_key, pose)

        if depth is not None:
            depth_key = f"{loc_key}-{i:0>2}_depth".encode("ascii")
            depth[depth > MAX_DEPTH] = 0.0
            # save depth in mm as uint16
            depth = (depth * 1000).astype(np.uint16)
            depth = zlib.compress(depth.tobytes())
            out_txn.put(depth_key, depth)

    out_txn.commit()


def save_data_per_pose_batch(db, data, poses, keys):
    """
    db: lmdb environment
    data: render results for poses at a location
    """
    out_txn = db.begin(write=True)
    for key, pose, rgb, depth in zip(keys, poses, data["colors"], data["depth"]):
        rgb_key = f"{key}_rgb".encode("ascii")
        rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        out_txn.put(rgb_key, rgb)

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
    use_locations = bool(args.locations_folder)
    use_loc_graph = bool(args.locations_graph_folder)
    use_poses = bool(args.poses_folder)
    assert (
        sum([use_locations, use_loc_graph, use_poses]) == 1
    ), "Need exact one of 3 format"
    os.makedirs(args.output_folder, exist_ok=True)
    scene_id = args.scene_id
    os.makedirs(args.error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.error_folder)]
    if scene_id in error_scenes:
        exit(0)
    try:
        # if True:
        if use_locations:
            print("run location")
            render_scene_from_location(scene_id, args, seed=args.seed)
        if use_poses:
            render_scene_from_pose(scene_id, args, seed=args.seed)
        if use_loc_graph:
            render_scene_from_location_graph(scene_id, args, seed=args.seed)
    except Exception as e:
        with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
            f.write(str(e))
        print(f"Error in scene {scene_id}")
"""
d71

CUDA_VISIBLE_DEVICES=0 blenderproc run \
examples/datasets/front_3d_with_improved_mat/render.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/work/vig/hieu/3dfront_data/images_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene \
--error_folder /work/vig/hieu/3dfront_data/images_100scenes_pano_error \
--end 100 \
--offset 0 \
--step 8
"""
