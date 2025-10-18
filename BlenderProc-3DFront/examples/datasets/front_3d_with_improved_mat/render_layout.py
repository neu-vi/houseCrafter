import blenderproc as bproc
import blenderproc.python.renderer.RendererUtility as RendererUtility
from blenderproc.python.types.MeshObjectUtility import create_with_empty_mesh

print()
import argparse
import json
import math
import os
import pickle
import zlib
from collections import Counter
from typing import List

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
    parser.add_argument("output_folder", nargs="?")

    parser.add_argument("--locations_graph_folder", type=str, default="")
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--layout_folder", type=str, required=True)
    parser.add_argument("--error_folder", type=str, default="./layout_error")

    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--n_heading", type=int, default=5)
    parser.add_argument("--elevation", type=float, default=40)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument(
        "--save_pose", type={"True": True, "False": False}.get, default=True
    )

    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()


def setup_box_data(args, scene_id, reverse=False):
    """
    if reverse , change the order of the vertices to render the back side of the box
    by making the front side having backface normal
    """
    layout_file = os.path.join(args.layout_folder, f"{scene_id}.json")
    furniture_layouts = json.load(open(layout_file))
    floor_height = furniture_layouts["floor_height"]
    ceiling_height = furniture_layouts["ceiling_height"]
    mesh_objects = []

    mesh_id = -1
    for box in furniture_layouts["boxes"]:
        mesh_id += 1
        obj = create_with_empty_mesh(box["model_jid"], box["instanceid"])
        mesh_objects.append(obj)

        obj.set_cp("uid", box["model_uid"])
        obj.set_cp("jid", box["model_jid"])
        obj.set_cp("inst_mark", "furniture_" + str(mesh_id))
        obj.set_cp("is_3D_future", True)
        obj.set_cp("category_id", box["label_id"])

        # make mesh for the box
        # while the a only have 4 corners, the following code works for any polygon
        # we assume the up direction is y-axis and the points follows the positive rotation of the y-axis
        vertices_xz = np.array(box["bbox"])  # n,2
        n = vertices_xz.shape[0]
        floor_vertices = np.stack(
            [vertices_xz[:, 0], np.ones(n) * floor_height, vertices_xz[:, 1]], axis=1
        )
        ceiling_vertices = np.stack(
            [vertices_xz[:, 0], np.ones(n) * ceiling_height, vertices_xz[:, 1]], axis=1
        )
        vertices = np.concatenate([ceiling_vertices, floor_vertices], axis=0)
        faces = []
        for i in range(n):
            faces.append([i, i + n, (i + 1) % n])
            faces.append([(i + 1) % n, i + n, (i + 1) % n + n])
        faces = np.array(faces)
        if reverse:
            faces = faces[:, [0, 2, 1]]

        # set the mesh
        mesh = obj.get_mesh()
        mesh.vertices.add(2 * n)
        mesh.vertices.foreach_set("co", vertices.flatten())
        mesh.loops.add(3 * 2 * n)  # 3 vertices per face and 2n faces
        mesh.loops.foreach_set("vertex_index", faces.flatten())
        mesh.polygons.add(2 * n)
        loop_start = np.arange(0, 3 * 2 * n, 3)
        loop_total = [3] * 2 * n
        mesh.polygons.foreach_set("loop_start", loop_start)
        mesh.polygons.foreach_set("loop_total", loop_total)
        mesh.update()
        obj.hide()
    return mesh_objects


def setup_house_structure_data(args, scene_id):
    """
    make mesh for house structure, (wall floor ceiling)

    """
    mapping_file = bproc.utility.resolve_resource(
        os.path.join("front_3D", "blender_label_mapping.csv")
    )
    label_mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    data = json.load(open(os.path.join(args.front_folder, f"{scene_id}.json")))

    mesh_objects = []
    mesh_id = -1
    for mesh_data in data["mesh"]:
        obj_name = mesh_data["type"].strip()
        if obj_name == "":
            obj_name = "void"

        obj = create_with_empty_mesh(obj_name, obj_name + "_mesh")
        mesh_objects.append(obj)
        mesh_id += 1
        obj.set_cp("uid", mesh_data["uid"])
        obj.set_cp("jid", mesh_data["jid"])
        obj.set_cp("inst_mark", "layout_" + str(mesh_id))
        obj.set_cp("is_3D_future", True)
        obj.set_cp("category_id", label_mapping.id_from_label(obj_name.lower()))

        # extract the vertices from the mesh_data
        vertices = np.array([float(ele) for ele in mesh_data["xyz"]])
        # extract the faces from the mesh_data
        faces = mesh_data["faces"]
        num_vertices = int(len(vertices) / 3)
        mesh = obj.get_mesh()
        mesh.vertices.add(num_vertices)
        mesh.vertices.foreach_set("co", vertices)

        # link the faces as vertex indices
        num_vertex_indicies = len(faces)
        mesh.loops.add(num_vertex_indicies)
        mesh.loops.foreach_set("vertex_index", faces)

        # the loops are set based on how the faces are a ranged
        num_loops = int(num_vertex_indicies / 3)
        mesh.polygons.add(num_loops)
        # always 3 vertices form one triangle
        loop_start = np.arange(0, num_vertex_indicies, 3)
        # the total size of each triangle is therefore 3
        loop_total = [3] * num_loops
        mesh.polygons.foreach_set("loop_start", loop_start)
        mesh.polygons.foreach_set("loop_total", loop_total)
        mesh.update()
        obj.hide()
    return mesh_objects


def setup_blender(args):
    bproc.init()
    RendererUtility.set_noise_threshold(0)
    RendererUtility.set_denoiser(None)
    RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
    RendererUtility.set_max_amount_of_samples(1)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    RendererUtility.set_cpu_threads(args.cpu_threads)

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


def render_scene_from_location(scene_id, args):
    POSES_PER_LOCATION = 12
    setup_blender(args)
    locations = np.load(os.path.join(args.locations_folder, f"{scene_id}.npy"))
    locations = {f"{i:0>4}": loc for i, loc in enumerate(locations)}

    # get invalid locations
    invalid_locations_path = os.path.join(
        args.locations_folder, f"{scene_id}_invalid.json"
    )
    if os.path.exists(invalid_locations_path):
        with open(invalid_locations_path, "r") as f:
            invalid_locations = set(json.load(f))
    else:
        invalid_locations = set()

    # get existing locations
    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode() for key in exist_keys]
        exist_keys = [key for key in exist_keys if not key.endswith("_pose")]
        exist_keys = [key.split("_")[0] for key in exist_keys]
        key_counters = Counter(exist_keys)
        exist_keys = [
            key for key, count in key_counters.items() if count == POSES_PER_LOCATION
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

    # load data
    house_structure = setup_house_structure_data(args, scene_id)
    box_data = setup_box_data(args, scene_id)

    for key, loc in tqdm(locations.items()):
        poses = get_camera_poses(loc)
        keys = [f"{key}_{i:0>2}" for i in range(len(poses))]
        render_batch(args.save_pose, env, house_structure, box_data, poses, keys)


def render_scene_from_location_graph(scene_id, args):
    POSES_PER_LOCATION = args.n_heading * 2
    setup_blender(args)
    location_graph = pickle.load(
        open(os.path.join(args.locations_graph_folder, f"{scene_id}.pkl"), "rb")
    )
    keys = sorted(list(location_graph.nodes))
    locations = {key: np.array(location_graph.nodes[key]["pos"]) for key in keys}

    # get existing locations
    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode() for key in exist_keys]
        exist_keys = [key for key in exist_keys if not key.endswith("_pose")]
        exist_keys = [key.split("-")[0] for key in exist_keys]
        key_counters = Counter(exist_keys)
        exist_keys = [
            key for key, count in key_counters.items() if count == POSES_PER_LOCATION
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

    # load data
    house_structure = setup_house_structure_data(args, scene_id)
    box_data = setup_box_data(args, scene_id)

    for key, loc in tqdm(locations.items()):
        poses = get_camera_poses2(loc, args.n_heading, args.elevation)
        keys = [f"{key}-{i:0>2}" for i in range(len(poses))]
        render_batch(args.save_pose, env, house_structure, box_data, poses, keys)


def render_scene_from_pose(scene_id, args, batch_size=12):
    setup_blender(args)
    poses = np.load(os.path.join(args.poses_folder, f"{scene_id}.npy"))
    poses = {f"{i:0>4}": loc for i, loc in enumerate(poses)}

    # get invalid poses
    invalid_poses_path = os.path.join(args.poses_folder, f"{scene_id}_invalid.json")
    if os.path.exists(invalid_poses_path):
        with open(invalid_poses_path, "r") as f:
            invalid_poses = set(json.load(f))
    else:
        invalid_poses = set()

    # get existing poses
    db_dir = os.path.join(args.output_folder, scene_id)
    os.makedirs(db_dir, exist_ok=True)
    env = lmdb.open(db_dir, map_size=int(1e12))
    with env.begin() as txn:
        exist_keys = list(txn.cursor().iternext(values=False))
        exist_keys = [key.decode() for key in exist_keys]
        exist_keys = [key for key in exist_keys if not key.endswith("_pose")]
        exist_keys = set(exist_keys)

    # filter invalid poses, skip computed poses
    for key in invalid_poses:
        poses.pop(key, None)
    for key in exist_keys:
        poses.pop(key, None)

    if not len(poses):
        print("scene {scene_id} completed.")
        return
    else:
        print(f"scene {scene_id} has {len(poses)} poses left to render.")

    # load data
    house_structure = setup_house_structure_data(args, scene_id)
    box_data = setup_box_data(args, scene_id)

    keys = list(poses.keys())
    n_batches = int(math.ceil(len(keys) / batch_size))
    for i in range(n_batches):
        batch_keys = keys[i * batch_size : (i + 1) * batch_size]
        batch_poses = [poses[key] for key in batch_keys]
        render_batch(
            args.save_pose, env, house_structure, box_data, batch_poses, batch_keys
        )


def render_batch(save_pose, env, house_structure, box_data, poses, keys):
    # render cls, depth from house structure
    for obj in house_structure:
        obj.hide(False)
    bproc.utility.reset_keyframes()
    for pose in poses:
        bproc.camera.add_camera_pose(pose)
    wall_depths = bproc.renderer.render()["depth"]
    wall_classes = bproc.renderer.render_segmap(map_by=["class"])["class_segmaps"]

    # render box data
    for obj in house_structure:
        obj.hide()
    furniture_depths: List[List[np.ndarray]] = [[] for _ in range(len(poses))]
    furniture_classes: List[List[int]] = [[] for _ in range(len(poses))]
    for obj in box_data:
        bproc.utility.reset_keyframes()
        obj.hide(False)
        for pose in poses:
            bproc.camera.add_camera_pose(pose)
        f_depths = bproc.renderer.render()["depth"]

        for w_depth, depth, f_depth_list, f_cls_list in zip(
            wall_depths, f_depths, furniture_depths, furniture_classes
        ):
            # only keep the part that is not occluded by the wall
            keep_mask = (depth < w_depth) & (depth < MAX_DEPTH)
            if not keep_mask.any():
                continue
            depth *= keep_mask
            f_depth_list.append((depth * 1000).astype(np.uint16))
            f_cls_list.append(obj.get_cp("category_id"))
        obj.hide()

    # post process data and save
    for frame_key, pose, w_depth, wall_cls, f_depth_list, f_cls_list in zip(
        keys, poses, wall_depths, wall_classes, furniture_depths, furniture_classes
    ):
        result_per_frame = post_process_data(
            w_depth, wall_cls, f_depth_list, f_cls_list
        )
        pose = pose if save_pose else None
        save_data_per_pose(env, result_per_frame, frame_key, pose)


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


def save_data_per_pose(db, data, key, pose):
    out_txn = db.begin(write=True)
    if pose is not None:
        pose_key = f"{key}_pose".encode("ascii")
        out_txn.put(pose_key, zlib.compress(pose.astype(np.float32).tobytes()))
    key = key.encode("ascii")
    out_txn.put(key, zlib.compress(data.tobytes()))
    out_txn.commit()


def post_process_data(w_depth, wall_cls, f_depth_list, f_cls_list):
    """
    sort intersections for each ray by label to get max number of intersections for each ray

    args:
        w_depth: H,w, in meter
        wall_cls: H,w, label of wall
        f_depth_list: list if H,W in mm
        f_cls_list: list of label

    return
        intersection info n,2,h,w: label and depth of intersection points for each ray
        NOTE: n is the maximum number of intersection points
            0 mean no intersection
    """
    # process wall data
    keep_mask = w_depth < MAX_DEPTH
    w_depth *= keep_mask
    w_depth = (w_depth * 1000).astype(np.uint16)  # mm
    wall_cls *= keep_mask
    wall_cls = wall_cls.astype(np.uint16)
    w_data = np.stack([wall_cls, w_depth], axis=0)[np.newaxis, ...]  # 1,2,h,w

    # process furniture data
    if not len(f_depth_list):
        return w_data
    f_depth = np.stack(f_depth_list, axis=0)  # n, h, w
    f_cls = np.array(f_cls_list, dtype=np.uint16).reshape(-1, 1, 1)  # n,1,1
    f_cls = f_cls * (f_depth > 100.0)  # n,h,w
    n_max = (f_cls > 0).sum(axis=0).max()
    if n_max == 0:
        return w_data

    # sort by label
    indices = np.flip(np.argsort(f_cls, axis=0), axis=0)  # n,h,w
    indices = indices[:n_max]  # n_max,h,w
    f_cls = np.take_along_axis(f_cls, indices, axis=0)
    f_depth = np.take_along_axis(f_depth, indices, axis=0)
    f_data = np.stack([f_cls, f_depth], axis=1)  # n_max,2,h,w

    # combine wall and furniture data into a single array
    out = np.concatenate([w_data, f_data], axis=0)  # n_max,2,h,w
    return out


if __name__ == "__main__":
    args = parse_args()
    use_locations = bool(args.locations_folder)
    use_loc_graph = bool(args.locations_graph_folder)
    use_poses = bool(args.poses_folder)
    assert (
        sum([use_locations, use_loc_graph, use_poses]) == 1
    ), "Need exact one of 3 format"
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.error_folder)]
    scene_id = args.scene_id
    if scene_id in error_scenes:
        exit(0)
    try:
        # if True:
        if use_locations:
            print("run location")
            render_scene_from_location(scene_id, args)
        if use_poses:
            render_scene_from_pose(scene_id, args)
        if use_loc_graph:
            render_scene_from_location_graph(scene_id, args)
    except Exception as e:
        with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
            f.write(str(e))
        print(f"Error in scene {scene_id}")
