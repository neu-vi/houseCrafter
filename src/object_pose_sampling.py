import argparse
import hashlib
import os
import pickle
from typing import List
import math
import numpy as np
import shapely
from shapely.geometry import Polygon
from tqdm import tqdm


def get_poses(house, rname, fname, boxes, seed, cfg, batch_size=20, margin=0.2):
    ceil, floor = house.get_ceil_floor_coords()
    room = house.rooms[rname]
    room_floor_polygons = room.floor_polygons

    # stack boxes
    furniture_names = sorted([name for name in boxes if name in room.children])
    box_centers, box_side_vectors, box_sizes = [], [], []
    for name in furniture_names:
        center, side_vector, size = boxes[name]
        box_centers.append(center)
        box_side_vectors.append(side_vector)
        box_sizes.append(size)
    box_centers = np.stack(box_centers)
    box_side_vectors = np.stack(box_side_vectors)

    rg = np.random.default_rng(seed)
    obj_box_center, obj_box_side_vector, obj_box_size = boxes[fname]

    valid_poses = []
    count = 0
    min_dist = cfg.min_dist + obj_box_size.max()
    min_dist = min(min_dist, ceil - floor - obj_box_size.min() - 0.5)
    max_dist = max(cfg.max_dist, obj_box_size.max() * cfg.max_rel_dist)
    desire_dist = np.sort(obj_box_size)[1:].sum()
    # desire_dist = min(desire_dist, ceil - floor - box_sizes[o_ind].min() - 0.1)
    while len(valid_poses) < cfg.n_pose:
        count += 1
        if count > 20:
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
        box_center = obj_box_center
        cam_center = box_center - look * distance
        up = np.array([[0, 1, 0]] * batch_size)
        poses = get_camera_pose(cam_center, look, up).astype(np.float32)

        # check height
        mask = (poses[:, 1, 3] < (ceil - margin)) & (poses[:, 1, 3] > (floor + margin))
        poses = poses[mask]
        if not len(poses):
            continue
        # check in room floor
        mask = point_in_room(room_floor_polygons, poses[:, :3, 3])
        poses = poses[mask]
        if not len(poses):
            continue
        # check out of the object box
        mask = point_out_of_boxes(box_centers, box_side_vectors, poses[:, :3, 3])
        poses = poses[mask]
        if not len(poses):
            continue
        if isinstance(valid_poses, list):
            valid_poses = poses
        else:
            valid_poses = np.concatenate([valid_poses, poses], axis=0)

    return valid_poses[: cfg.n_pose]


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


def match_poses(src_poses, src_kdtree, target_poses, angle=30, distance=0.3):
    """
    given set of src poses and target poses
    find a subset of src poses where each pose has high overlap with at least one target pose

    'overlap' is chosen based on distance and angle

    args:
        src_poses: n,4,4 c2w matrix
        target_poses: m,4,4 c2w matrix
        angle: degree, the angle threshold
        distance: meter, the distance threshold
        src_kdtree: KDTree built from src poses locations
    return
        src_pose_ids: m, list of indices of src poses m <= n
    NOTE: assume cam forward direction is -z
    """
    cos_thr = np.cos(np.radians(angle))
    out_ids = []
    src_look = -src_poses[:, :3, 2]
    for t_pose in target_poses:
        indices = src_kdtree.query_radius(t_pose[:3, 3][None], r=distance)[0]
        if not len(indices):
            continue
        candidates = src_look[indices]
        look = -t_pose[:3, 2]
        cos = np.einsum("ni, i -> n", candidates, look)
        ids = indices[cos > cos_thr].tolist()
        out_ids.extend(ids)
    out_ids = np.array(list(set(out_ids)))
    return out_ids

def pose_interpolation(pose1, pose2, dist_thr=0.3, angle_thr=30.0):
    """
    interpolate between 2 poses based on distance and look vector
    meaning, the intermediate poses are selected by interpolate the camera center 
    and the look vector
    
    while the up vector of the camera is in the plane 
    containing look vector and up direction of the scene
    NOTE: assuming the up vector of the scene is y-axis
        and camera coordinate is x right, y up, z backward
        
    args:
        pose1, pose2: nparray c2w 4x4
    return intermediate poses in the order from pose1 to pose2 (not include pose1, pose2)
        n,4,4
    """
    center2 = pose2[:3, 3] 
    center1 = pose1[:3, 3]
    n_poses_dist = math.ceil(np.linalg.norm(center2-center1)/dist_thr) - 1
    look1 = -pose1[:3, 2]
    look2 = -pose2[:3, 2]
    angle_diff = np.arccos(np.dot(look1, look2))
    n_poses_angle = math.ceil(angle_diff/np.radians(angle_thr)) - 1
    n_poses = max(n_poses_dist, n_poses_angle)
    if n_poses < 1:
        return np.ndarray(shape=(0,4,4))
    weights = np.linspace(0, 1, n_poses+1, endpoint=False)[1:, np.newaxis]
    looks = look1*(1-weights) + look2*weights
    centers = center1*(1-weights) + center2*weights
    
    up = np.array([[0, 1, 0]] * n_poses)
    return get_camera_pose(centers, looks, up).astype(np.float32)