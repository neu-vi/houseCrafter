"""
make

"""

import argparse

from sklearn.neighbors import KDTree
import numpy as np
import networkx as nx

# import open3d as o3d
import quaternion as qt
import os
from tqdm import tqdm
import pickle


class PoseGraph:
    K = np.array([[256.0, 0, 256], [0, 256, 256], [0, 0, 1]], dtype=np.float64)

    @staticmethod
    def make_edges(poses, distance, angle):
        """
        2 pose is connected if they are within distance and angle
        camera coordinate is x right, y up, z backward
        """
        cosine_thr = np.cos(np.radians(angle))
        # make graph based on distance only
        kdtree = KDTree(poses[:, :3, 3], leaf_size=30, metric="euclidean")

        # forward direction of the pose
        ## came forward is -z
        cam_forward = np.array([0, 0, -1])
        world_forward = np.einsum("nij, j->ni", poses[:, :3, :3], cam_forward)

        edges = set()
        # select edge based angle
        for i in range(poses.shape[0] - 1):
            # find nearest neighbor
            neighbor_indices = kdtree.query_radius(poses[i : i + 1, :3, 3], r=distance)[
                0
            ]
            # only need to check for index > i to reduce redundancy
            neighbor_indices = neighbor_indices[neighbor_indices > i]
            if not len(neighbor_indices):
                continue
            # check angle between forward direction
            neighbor_directions = world_forward[neighbor_indices]
            cosines = np.einsum("ni, i -> n", neighbor_directions, world_forward[i])
            selected_indices = neighbor_indices[cosines >= cosine_thr]
            for j in selected_indices:
                edges.add((i, j))
                edges.add((j, i))
        return edges

    @staticmethod
    def make_graph(poses, edges):
        G = nx.Graph()
        for i, pose in enumerate(poses):
            G.add_node(f"{i:0>4}", pose=pose)
        for i, j in edges:
            G.add_edge(f"{i:0>4}", f"{j:0>4}")
        return G

    @classmethod
    def show_G(cls, G):
        nodes = sorted(list(G.nodes))
        for i, n in enumerate(nodes):
            assert i == int(n)
        poses = np.array([G.nodes[n]["pose"] for n in nodes])
        edges = list(G.edges)
        edges = [(nodes.index(i), nodes.index(j)) for i, j in edges]
        cls.show(poses, edges)

    @classmethod
    def show(cls, poses, edges):
        try:
            import open3d as o3d
        except ImportError:
            print("open3d is not installed")
            return
        poses[:, 3, 3] = 1.0
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(poses[:, :3, 3])

        o3d_lines = o3d.geometry.LineSet()
        o3d_lines.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
        lines = np.array(list(edges))
        o3d_lines.lines = o3d.utility.Vector2iVector(lines)

        cameras = []

        w2c = np.linalg.inv(poses.astype(np.float64))
        for p in w2c:
            camera = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=512,
                view_height_px=512,
                intrinsic=cls.K,
                extrinsic=p,
                scale=0.05,
            )
            cameras.append(camera)
        o3d.visualization.draw_geometries([o3d_pcd, o3d_lines] + cameras)

    # TODO filter edges based on depth:
    """
    camera pose is just a heuristic to decide if 2 view has high overlap or not
    the better way to do so is to use depth map, where we can get the pointcloud of 2 views
    and calculate the overlap based on the correspondence of the pointcloud
    """

    def filter_edges(self, poses, edges, house):
        pass

    @staticmethod
    def save_graph(G, path):
        pickle.dump(G, open(path, "wb"))
        # nx.write_gpickle(G, path)


class LocationGraph(PoseGraph):
    @staticmethod
    def make_edges(locations, distance):
        """
        2 location is connected if they are within distance
        """
        # make graph based on distance only
        kdtree = KDTree(locations, leaf_size=30, metric="euclidean")

        # forward direction of the pose
        ## came forward is -z
        # cam_forward = np.array([0, 0, -1])
        # world_forward = np.einsum('nij, j->ni', poses[:, :3, :3], cam_forward)

        edges = set()
        # select edge based angle
        for i in range(locations.shape[0] - 1):
            # find nearest neighbor
            neighbor_indices = kdtree.query_radius(locations[i : i + 1], r=distance)[0]
            # only need to check for index > i to reduce redundancy
            neighbor_indices = neighbor_indices[neighbor_indices > i]
            if not len(neighbor_indices):
                continue
            for j in neighbor_indices:
                edges.add((i, j))
                edges.add((j, i))
        return edges

    @classmethod
    def show_G(cls, G):
        nodes = sorted(list(G.nodes))
        for i, n in enumerate(nodes):
            assert i == int(n)
        locations = np.array([G.nodes[n]["location"] for n in nodes])
        edges = list(G.edges)
        edges = [(nodes.index(i), nodes.index(j)) for i, j in edges]
        cls.show(locations, edges)

    @classmethod
    def show(cls, locations, edges):
        try:
            import open3d as o3d
        except ImportError:
            print("open3d is not installed")
            return
        # visualize
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(locations)

        o3d_lines = o3d.geometry.LineSet()
        o3d_lines.points = o3d.utility.Vector3dVector(locations)
        lines = np.array(list(edges))
        o3d_lines.lines = o3d.utility.Vector2iVector(lines)

        # make camera
        K = np.array([[256.0, 0, 256], [0, 256, 256], [0, 0, 1]], dtype=np.float64)
        cameras = []

        for location in locations:
            poses = cls.get_camera_poses(location)
            w2c = np.linalg.inv(poses.astype(np.float64))
            for p in w2c:
                camera = o3d.geometry.LineSet.create_camera_visualization(
                    view_width_px=512,
                    view_height_px=512,
                    intrinsic=K,
                    extrinsic=p,
                    scale=0.05,
                )
                cameras.append(camera)

        o3d.visualization.draw_geometries([o3d_pcd, o3d_lines] + cameras)

    @staticmethod
    def make_graph(locations, edges):
        G = nx.Graph()
        for i, location in enumerate(locations):
            G.add_node(f"{i:0>4}", location=location)
        for i, j in edges:
            G.add_edge(f"{i:0>4}", f"{j:0>4}")
        return G

    @classmethod
    def get_camera_poses(cls, location: np.ndarray) -> np.ndarray:
        """
        copied from render.py
        select poses from the given location

        try 2 settings
            12 views, 0 deg elevation
            12 views, 6 look down 30 deg 6 look up 30 deg

        NOTE: assume y is up
        # cam coor in blender is x right, y up, z backward

        """
        HEADINGS = np.arange(12) * 2 * np.pi / 12
        # rotation for setting 1
        ROTATIONS1 = qt.from_rotation_vector(
            HEADINGS[:, np.newaxis] * np.array([0, 1, 0])
        )
        ROTATIONS1 = qt.as_rotation_matrix(ROTATIONS1)

        poses = np.zeros((len(HEADINGS), 4, 4), dtype=np.float32)
        poses[:, :3, :3] = ROTATIONS1
        poses[:, :3, 3] += location
        poses[:, 3, 3] = 1
        return poses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--dist", type=float)
    parser.add_argument("--angle", type=float, default=60)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=7000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    return parser.parse_args()


def make_pose_graph(scene_id, args):
    poses = np.load(os.path.join(args.poses_folder, f"{scene_id}.npy"))
    if not len(poses):
        print(f"Empty pose for {scene_id}")
        return
    edges = PoseGraph.make_edges(poses, args.dist, args.angle)
    G = PoseGraph.make_graph(poses, edges)
    PoseGraph.save_graph(G, os.path.join(args.output_folder, f"{scene_id}.pkl"))


def make_location_graph(scene_id, args):
    locations = np.load(os.path.join(args.locations_folder, f"{scene_id}.npy"))
    edges = LocationGraph.make_edges(locations, args.dist)
    G = LocationGraph.make_graph(locations, edges)
    LocationGraph.save_graph(G, os.path.join(args.output_folder, f"{scene_id}.pkl"))


if __name__ == "__main__":
    args = parse_args()
    assert args.locations_folder or args.poses_folder, "Need locations or poses folder"
    assert not (
        args.locations_folder and args.poses_folder
    ), "Only one of locations or poses folder should be provided"

    os.makedirs(args.output_folder, exist_ok=True)
    if args.locations_folder:
        scene_ids = sorted(os.listdir(args.locations_folder))
        scene_ids = [f.split(".")[0] for f in scene_ids if f.endswith(".npy")]
        scene_ids = scene_ids[args.start : args.end]
        scene_ids = scene_ids[args.offset :: args.step]
        for scene_id in tqdm(scene_ids):
            make_location_graph(scene_id, args)
    if args.poses_folder:
        scene_ids = sorted(os.listdir(args.poses_folder))
        scene_ids = [f.split(".")[0] for f in scene_ids if f.endswith(".npy")]
        scene_ids = scene_ids[args.start : args.end]
        scene_ids = scene_ids[args.offset :: args.step]
        for scene_id in tqdm(scene_ids):
            make_pose_graph(scene_id, args)

    # G = nx.read_gpickle(
    #     "/media/hieu/T7/3d-front/graph_poses_100_2/00ad8345-45e0-45b3-867d-4a3c88c2517a.pkl"
    # )
    # PoseGraph.show_G(G)

    # G = nx.read_gpickle(
    #     "/media/hieu/T7/3d-front/graph_locations_100_7/00ad8345-45e0-45b3-867d-4a3c88c2517a.pkl"
    # )
    # LocationGraph.show_G(G)
"""
random pose on floor dist xz 0.2, y 1.2-1.7: dist 0.5, angle 60

pano view from location dist xz 0.7, y 1.2-1.7: dist 1.0

"""
