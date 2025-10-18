import argparse
import hashlib
import os
import pickle
from typing import List

import numpy as np
import shapely
from front3d_scene import load_scene
from shapely.geometry import Polygon
from tqdm import tqdm


def hash_instance_id(instance_id):
    return int(hashlib.sha1(instance_id.encode("utf-8")).hexdigest(), 16) % (10**8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_folder", type=str, help="Path to the 3D front file")
    parser.add_argument(
        "--future_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument(
        "--future_bbox_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument(
        "--min_dist", type=float, default=0.2, help="minimum distance to object center"
    )
    parser.add_argument(
        "--max_dist", type=float, default=2.0, help="maximum distance to object center"
    )
    parser.add_argument(
        "--max_rel_dist",
        type=float,
        default=2.0,
        help="maximum relative distance to object center, this will be multiply with largest side of the object",
    )
    parser.add_argument(
        "--pose_per_obj", type=int, default=100, help="number of poses per object"
    )
    parser.add_argument("--start", type=int, default=0, help="Field of view of camera.")
    parser.add_argument("--error_folder", type=str)
    parser.add_argument(
        "--end", type=int, default=7000, help="Field of view of camera."
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Field of view of camera."
    )
    parser.add_argument("--step", type=int, default=1, help="Field of view of camera.")
    return parser.parse_args()


def get_poses(scene_id, args, seed, batch_size=100):
    scene_file = os.path.join(args.front_folder, f"{scene_id}.json")
    house = load_scene(scene_file, args.future_folder, args.future_bbox_folder)
    ceil, floor = house.get_ceil_floor_coords()
    room_names = sorted(house.rooms.keys())

    all_poses_data = {}
    for room_name in room_names:
        room_floor_polygons = house.rooms[room_name].floor_polygons
        if not len(room_floor_polygons):
            continue
        furniture_names = sorted(
            [
                name
                for name, f in house.rooms[room_name].children.items()
                if f.is_furniture()
            ]
        )
        if not len(furniture_names):
            continue
        box_centers, box_side_vectors, box_sizes = [], [], []
        for fname in furniture_names:
            f = house.rooms[room_name].children[fname]
            bbox = f.get_bbox()
            center, side_vector = bbox.get_center_side_vector()
            box_centers.append(center)
            box_side_vectors.append(side_vector)
            box_sizes.append(bbox.size)
        box_centers = np.stack(box_centers)
        box_side_vectors = np.stack(box_side_vectors)

        for o_ind, fname in enumerate(furniture_names):
            obj_seed = hash_instance_id(fname) + seed
            rg = np.random.default_rng(obj_seed)

            valid_poses = []
            count = 0
            min_dist = args.min_dist + box_sizes[o_ind].max()
            min_dist = min(min_dist, ceil - floor - box_sizes[o_ind].min() - 0.5)
            max_dist = max(args.max_dist, box_sizes[o_ind].max() * args.max_rel_dist)
            desire_dist = np.sort(box_sizes[o_ind])[1:].sum()
            # desire_dist = min(desire_dist, ceil - floor - box_sizes[o_ind].min() - 0.1)
            while len(valid_poses) < args.pose_per_obj:
                count += 1
                if count > 10:
                    break
                elevation = np.radians(rg.uniform(-89, 89, size=(batch_size,)))
                heading = rg.uniform(-np.pi, np.pi, size=(batch_size,))

                if desire_dist < min_dist:
                    distance = rg.uniform(min_dist, max_dist, size=(batch_size, 1))
                else:
                    var = min(max_dist - desire_dist, desire_dist - min_dist)
                    loc = desire_dist * 0.95**count
                    distance = rg.normal(loc, var, size=(batch_size, 1))
                distance = np.clip(distance, min_dist, max_dist)

                look = get_direction(elevation, heading)
                box_center = box_centers[o_ind]
                cam_center = box_center - look * distance
                up = np.array([[0, 1, 0]] * batch_size)
                poses = get_camera_pose(cam_center, look, up).astype(np.float32)

                # check height
                mask = (poses[:, 1, 3] < ceil) & (poses[:, 1, 3] > floor)
                poses = poses[mask]
                if not len(poses):
                    continue
                # check in room floor
                mask = point_in_room(room_floor_polygons, poses[:, :3, 3])
                poses = poses[mask]
                if not len(poses):
                    continue
                # check out of the object box
                mask = point_out_of_boxes(
                    box_centers, box_side_vectors, poses[:, :3, 3]
                )
                poses = poses[mask]
                if not len(poses):
                    continue
                if isinstance(valid_poses, list):
                    valid_poses = poses
                else:
                    valid_poses = np.concatenate([valid_poses, poses], axis=0)

            if len(valid_poses):
                all_poses_data[fname] = {
                    "poses": valid_poses,
                    "room": room_name,
                }
    pose_path = os.path.join(args.poses_folder, f"{scene_id}.pkl")
    pickle.dump(all_poses_data, open(pose_path, "wb"))


def point_in_room(room_floor_polygons: List[Polygon], points):
    """
    polygons : list of shapely.geometry.Polygon
    points : np.array of shape (N, 3)
    NOTE: polygon are in xz plane
    return inliers: np.array of shape (N, ) boolean
    """
    xz = points[:, [0, 2]]
    points = shapely.points(xz)
    inliers = [f.contains(points) for f in room_floor_polygons]
    inliers = sum(inliers) > 0
    return inliers


def point_out_of_boxes(box_centers, box_side_vector, points):
    """
    each box is represented by center and side vectors
    side vector is the vector from center to to the face of the box
    box_centers: m, 3
    side_vector: m, 3, 3 # 3 vector for each box, last dim is xyz
    points: n, 3
    return inliers: np.array of shape (n, ) boolean
    """

    # center to point vector
    c2p = points[:, None] - box_centers  # n, m, 3

    # get projection of c2p to side vector
    projection = np.einsum("nmd,mbd->nmb", c2p, box_side_vector)  # n, m, 3
    projection = np.absolute(projection)
    box_side_vector_sq = np.linalg.norm(box_side_vector, axis=-1) ** 2
    outliers = np.any(projection > box_side_vector_sq, axis=-1)  # n, m
    outliers = np.all(outliers, axis=-1)  # n
    return outliers


def get_direction(elevation, heading):
    """
    return vector direction
    zero heading is along z axis

    args: elevation, heading (n,) in radians
    return (n, 3) vector
    """
    return np.stack(
        [
            np.cos(elevation) * np.sin(heading),
            np.sin(elevation),
            np.cos(elevation) * np.cos(heading),
        ],
        axis=-1,
    )


def get_camera_pose(center, look, up):
    """
    center: 3d point n,3
    look: direction vector n,3
    up: direction vector n,3
    return c2w matrix n,4,4
    NOTE: the given up vector is not necessary orthogonal to the look vector
    the correct up vector will be computed
    """
    right = np.cross(look, up)
    up = np.cross(right, look)
    look = look / np.linalg.norm(look, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    pose = np.zeros((right.shape[0], 4, 4), dtype=np.float32)
    pose[:, :3, 0] = right
    pose[:, :3, 1] = up
    pose[:, :3, 2] = -look
    pose[:, :3, 3] = center
    pose[:, 3, 3] = 1
    return pose


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.poses_folder, exist_ok=True)
    scene_ids = sorted(
        [f.split(".")[0] for f in os.listdir(args.front_folder) if f.endswith(".json")]
    )
    scene_id_dict = {i: scene_id for i, scene_id in enumerate(scene_ids)}
    scene_ids = scene_ids[args.start : args.end]
    scene_ids = set(scene_ids[args.offset :: args.step])
    scene_id_dict = {
        i: scene_id for i, scene_id in scene_id_dict.items() if scene_id in scene_ids
    }
    os.makedirs(args.error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.error_folder)]
    for index, scene_id in tqdm(scene_id_dict.items()):
        if os.path.exists(os.path.join(args.poses_folder, f"{scene_id}.npy")):
            continue
        if scene_id in error_scenes:
            continue
        try:
            # if True:
            get_poses(scene_id, args, seed=index)
        except Exception as e:
            with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
                f.write(str(e))
            print(f"Error in scene {scene_id}")
