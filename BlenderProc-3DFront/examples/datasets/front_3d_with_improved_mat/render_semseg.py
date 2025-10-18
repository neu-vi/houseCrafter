import blenderproc as bproc
import blenderproc.python.renderer.RendererUtility as RendererUtility

import argparse
import json
import math
import os
import zlib
from collections import Counter
import pickle

import lmdb
import numpy as np
from tqdm import tqdm

# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument(
        "front_3D_texture_folder", help="Path to the 3D FRONT texture folder."
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
    parser.add_argument("--error_folder", type=str, default="./error")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()


def check_name(name, category_name):
    return True if category_name in name.lower() else False


def setup_blender(args):
    bproc.init()
    RendererUtility.set_noise_threshold(0)
    RendererUtility.set_denoiser(None)
    RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
    RendererUtility.set_max_amount_of_samples(1)

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

    return loaded_objects


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
        exist_keys = [key for key, count in key_counters.items() if count == 4]

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
        default_values = {
            "location": [0, 0, 0],
            "cp_inst_mark": "",
            "cp_uid": "",
            "cp_jid": "",
            "cp_room_id": "",
        }
        data = bproc.renderer.render_segmap(
            map_by=[
                "instance",
                "class",
                "cp_uid",
                "cp_jid",
                "cp_inst_mark",
                "cp_room_id",
                "location",
            ],
            default_values=default_values,
        )
        save_data_per_pose_batch(env, data, batch_poses, batch_keys)


def save_data_per_pose_batch(db, data, poses, keys):
    """
    db: lmdb environment
    data: render results for poses at a location
    """
    out_txn = db.begin(write=True)
    for key, pose, inst_seg, sem_seg, attr in zip(
        keys,
        poses,
        data["instance_segmaps"],
        data["class_segmaps"],
        data["instance_attribute_maps"],
    ):

        pose_key = f"{key}_pose".encode("ascii")
        pose = zlib.compress(pose.astype(np.float32).tobytes())
        out_txn.put(pose_key, pose)

        inst_key = f"{key}_instance".encode("ascii")
        inst_seg = zlib.compress(inst_seg.tobytes())
        out_txn.put(inst_key, inst_seg)

        sem_key = f"{key}_sem".encode("ascii")
        sem_seg = zlib.compress(sem_seg.tobytes())
        out_txn.put(sem_key, sem_seg)

        attr_key = f"{key}_attr".encode("ascii")
        attr = pickle.dumps(attr)
        out_txn.put(attr_key, attr)
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
