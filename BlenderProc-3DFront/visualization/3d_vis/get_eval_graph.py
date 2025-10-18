from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
import open3d as o3d
import quaternion as qt
import torch
import torch.nn.functional as F
from front3d_scene import House, load_labels, load_scene
from scipy import ndimage
from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from sklearn.neighbors import KDTree
from torch import Tensor
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_folder", type=str, help="Path to the 3D front file")
    parser.add_argument(
        "--future_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument(
        "--future_bbox_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument("--graph_folder", type=str, default="")
    parser.add_argument("--error_folder", type=str)
    parser.add_argument(
        "--dist", type=float, default=1.0, help="distance of grid point in meter"
    )
    parser.add_argument(
        "--map_res", type=float, default=0.01, help="resolution of the map in meter"
    )
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--variance", type=float, default=0.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=7000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_graph(scene_id, args):
    scene_file = os.path.join(args.front_folder, f"{scene_id}.json")
    house = load_scene(scene_file, args.future_folder, args.future_bbox_folder)

    # get locations
    floor_polys = []
    furniture_polys = []
    for room in house.rooms.values():
        floor_polys.extend([np.asarray(x.exterior.coords) for x in room.floor_polygons])
        furniture_polys.extend([x["bbox"] for x in room.get_furniture_2dbox()])
    floor_map = FloorMap.from_polygons(floor_polys, furniture_polys)
    floor_map = floor_map.get_largest_connected_component()
    map_filler = MapFiller(
        torch.tensor(floor_map.map_array, device=args.device),
        grid_size=int(args.dist / args.map_res),
        min_size=10,
        min_area=100,
        radius=int(args.dist / args.map_res),
        variance=int(args.variance / args.map_res),
    )
    points, uncovered_mask = map_filler.iterative_fill()
    points_w = floor_map.points_in_world_coords(points.cpu().numpy())

    # partition locations into room
    room2point_ids = {}
    room2points = {}
    for r_name, room in house.rooms.items():
        inliers = room.points_in_room(points_w)
        if inliers is None:
            continue
        point_ids = np.where(inliers)[0]
        if len(point_ids) == 0:
            continue
        room2point_ids[r_name] = point_ids
        room2points[r_name] = points_w[point_ids]

    # build graph within each room
    room_graphs = {}
    for r_name, point_ids in room2point_ids.items():
        room_points = points_w[point_ids]
        keys = [f"{i:0>4}" for i in point_ids]
        room_graphs[r_name] = build_graph_in_room(
            args.dist + args.eps, room_points, keys, r_name
        )

    # connect between rooms
    door_extractor = DoorWindowExtractor(house)
    doors = door_extractor.extract_doors()

    G = nx.union_all(list(room_graphs.values()))
    for door in doors:
        if len(door.rooms) != 2:
            continue

        edge = []
        for room in door.rooms:
            assert room in room2point_ids, f"{room} has no points"
            point_ids = room2point_ids[room]
            dists = np.linalg.norm(
                points_w[point_ids] - door.bottom_vertices[:, [0, 2]].mean(axis=0),
                axis=1,
            )
            min_id = point_ids[np.argmin(dists)]
            edge.append(f"{min_id:0>4}")
        G.add_edge(*edge, category="door")
    nodes = list(G.nodes)

    # adding height
    heights = np.array([args.height] * len(nodes))
    if args.variance > 0:
        rg = np.random.default_rng(args.seed)
        heights += rg.uniform(-args.variance, args.variance, len(nodes))
    for node, height in zip(nodes, heights):
        pos = G.nodes[node]["pos"]
        G.nodes[node]["pos"] = np.array([pos[0], height, pos[1]])
    return G
    # save graph


def build_graph_in_room(
    dist_threshold: float, points: np.ndarray, keys: List[str], room_name: str
) -> nx.Graph:
    """
    build graph for locations in a single room
    make edge where distance between points is less than a threshold
    then connect the connected components
    """
    kdtree = KDTree(points, metric="euclidean")
    G = nx.Graph()
    for key, point in zip(keys, points):
        G.add_node(key, pos=point, room=room_name)

    adj = kdtree.query_radius(points, r=dist_threshold)
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            if i == j:
                continue
            G.add_edge(keys[i], keys[j], category="distance")
    connected_components = list(nx.connected_components(G))
    if len(connected_components) == 1:
        return G
    component_dist = {}
    for i in range(1, len(connected_components)):
        for j in range(i):
            dist = np.inf
            min_pair = None
            for k in connected_components[i]:

                for l in connected_components[j]:
                    current_dist = np.linalg.norm(
                        points[keys.index(k)] - points[keys.index(l)]
                    )
                    if current_dist < dist:
                        min_pair = (k, l)
                        dist = current_dist

            component_dist[(i, j)] = (dist, min_pair)

    # find mimimun spanning tree of the connected components
    component_graph = nx.Graph()
    for i in range(1, len(connected_components)):
        component_graph.add_node(i)
    for key, (dist, min_pair) in component_dist.items():
        component_graph.add_edge(*key, weight=dist)
    mst = nx.minimum_spanning_tree(component_graph)

    for edge in mst.edges:
        i, j = sorted(edge, reverse=True)
        k, l = component_dist[(i, j)][1]
        G.add_edge(k, l, category="connect")

    return G


class FloorMap:
    """
    rasterized map of free space on the floor
    region in floor polygon and out of furniture 2d box is free space

    map dimension:
        assume y the up direction in world coordinate
        the map represent the xz plane
        o -------> x
        |
        |
        |
        z
    attributes:
        resolution: float, resolution of the map in meter
        offset: (x,z) the world coordinate of the corner of the top-left pixel of the map
    """

    def __init__(self, map_array: np.ndarray, resolution: float, offset: np.ndarray):

        self.map_array = map_array
        self.resolution = resolution
        self.offset = offset

    @classmethod
    def from_polygons(
        cls,
        floor_polys: List[np.ndarray],
        furniture_polys: List[np.ndarray] = [],
        resolution=0.01,
    ) -> FloorMap:
        """
        poly: (n,2) x,z world coords of the polygon
        """
        assert len(floor_polys) > 0, "No floor polygon"
        min_x = min([p[:, 0].min() for p in floor_polys])
        min_z = min([p[:, 1].min() for p in floor_polys])
        max_x = max([p[:, 0].max() for p in floor_polys])
        max_z = max([p[:, 1].max() for p in floor_polys])

        min_x = math.floor(min_x / resolution) * resolution
        min_z = math.floor(min_z / resolution) * resolution
        max_x = math.ceil(max_x / resolution) * resolution
        max_z = math.ceil(max_z / resolution) * resolution
        offset = np.array([min_x, min_z])

        # true mean free space, false mean occupied
        map_array = np.zeros(
            (int((max_z - min_z) / resolution), int((max_x - min_x) / resolution)),
            dtype=np.uint8,
        )
        for poly in floor_polys:
            poly = np.round((poly - offset) / resolution).astype(np.int32)
            cv2.fillPoly(map_array, pts=[poly], color=(1))

        out = cls(map_array, resolution, offset)
        out.add_furnitures(furniture_polys)
        return out

    def add_furnitures(self, furniture_polys: List[np.ndarray]):
        if len(furniture_polys):
            # furniture_polys = [
            #     np.round((poly - self.offset) / self.resolution).astype(np.int32) for poly in furniture_polys
            # ]
            # cv2.fillPoly(self.map_array, pts=furniture_polys, color=(0))

            for poly in furniture_polys:
                poly = np.round((poly - self.offset) / self.resolution).astype(np.int32)
                min_coords = poly.min(axis=0)
                max_coords = poly.max(axis=0)
                skip = (
                    np.any(min_coords[0] < 0)
                    or (max_coords[0] >= self.map_array.shape[1])
                    or max_coords[1] >= self.map_array.shape[0]
                )
                if skip:
                    continue
                cv2.fillPoly(self.map_array, pts=[poly], color=(0))

    def get_largest_connected_component(self, iters=20) -> FloorMap:
        """
        return list of connected components
        """
        # closing gaps
        closing = binary_closing(self.map_array, iterations=iters)
        closing = self.map_array | closing
        # find largest connected component
        label_im, nb_labels = ndimage.label(closing)
        sizes = ndimage.sum(closing, label_im, range(nb_labels + 1))
        mask = label_im == sizes.argmax()

        props = regionprops(mask.astype(np.uint8))
        row_min, col_min, row_max, col_max = props[0].bbox
        map_array = self.map_array[row_min:row_max, col_min:col_max]
        offset = self.offset + np.array([col_min, row_min]) * self.resolution
        return FloorMap(map_array, self.resolution, offset)

    def points_in_world_coords(self, points: np.ndarray) -> np.ndarray:
        """
        points: (n,2) row,col index of points in the map

        return (n,2) x,z world coords of the points
        """
        return (points[:, [1, 0]] + 0.5) * self.resolution + self.offset

    def points_in_pixel_indices(self, points: np.ndarray) -> np.ndarray:
        """
        points: (n,2) x,z world coords of the points

        return (n,2) row,col index of points in the map
        """
        return np.round((points - self.offset) / self.resolution).astype(np.int32)[
            :, [1, 0]
        ]


class MapFiller:
    def __init__(
        self,
        floor_map: Tensor,
        grid_size: int = 100,
        min_size=10,
        radius=100,
        min_area=100,
        variance: int = 0,
        seed=0,
    ):
        """
        floor_map (h,w) binary mask, 1 is free space, 0 is occupied
        """
        self.floor_map = floor_map
        self.grid_size = grid_size
        self.radius = radius
        self.min_size = min_size
        self.min_area = min_area
        self.variance = variance
        self.rg = torch.Generator(device=floor_map.device).manual_seed(seed)

    def _fill_map(self, mask: Tensor, attemp=3) -> Tuple[Tensor]:
        """
        given a binary mask, fill the free space in the mask
        mask: (h,w) binary mask, 1 is free space, 0 is occupied

        return
            points: (n,2) row,col indices of the points
            uncovered_mask: (h,w) binary mask, 1 is uncovered, 0 is covered
        """
        h, w = mask.shape
        col_random = self.variance > 0 and w > self.min_size
        row_random = self.variance > 0 and h > self.min_size

        def _get_points(start_row, start_col):
            r = torch.arange(start_row, h, self.grid_size, device=mask.device)
            c = torch.arange(start_col, w, self.grid_size, device=mask.device)
            grid = torch.stack(
                torch.meshgrid(
                    r,
                    c,
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            if col_random:
                grid[:, 1] += torch.randint(
                    -self.variance,
                    self.variance + 1,
                    (grid.shape[0],),
                    generator=self.rg,
                    device=mask.device,
                )
            if row_random:
                grid[:, 0] += torch.randint(
                    -self.variance,
                    self.variance + 1,
                    (grid.shape[0],),
                    generator=self.rg,
                    device=mask.device,
                )
            grid.clamp_min_(0)
            grid[:, 0].clamp_max_(h - 1)
            grid[:, 1].clamp_max_(w - 1)
            return grid

        # errode the mask to get the uncovered area for points
        neg_mask = -mask[None, None].float()
        errode_size = self.min_size // 2
        for _ in range(errode_size):
            neg_mask = F.max_pool2d(neg_mask, kernel_size=3, stride=1, padding=1)
        erroded_mask = neg_mask[0, 0].bool()

        # initial start row, col
        start_row = (h % self.grid_size) // 2
        start_row = (
            start_row + self.grid_size // 2 if h >= self.grid_size else start_row
        )
        start_col = (w % self.grid_size) // 2
        start_col = (
            start_col + self.grid_size // 2 if w >= self.grid_size else start_col
        )

        # try some attemps to find the start row, col
        count = 0
        step_row = max(start_row // attemp, 1)
        step_col = max(start_col // attemp, 1)
        while count < attemp:
            points = _get_points(start_row, start_col)
            point_mask = erroded_mask[points[:, 0], points[:, 1]]
            if point_mask.sum() > 0:
                break
            start_row -= step_row
            start_col -= step_col
            if start_row < 0 or start_col < 0:
                break

        # erroded_mask = F.max_pool2d(-mask[None, None].float(), kernel_size=11, stride=1, padding=5)[0,0].bool()

        points = points[point_mask]
        covered_mask = torch.zeros_like(mask).to(dtype=torch.float)
        covered_mask[points[:, 0], points[:, 1]] = 1
        # covered_mask = F.max_pool2d(
        #     covered_mask[None, None],
        #     kernel_size=2 * self.radius + 1,
        #     stride=1,
        #     padding=self.radius,
        # )[0,0]
        for _ in range(self.radius):
            covered_mask = F.max_pool2d(
                covered_mask[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]

        uncovered_mask = mask & (~covered_mask.bool())
        return points, uncovered_mask

    def iterative_fill(self) -> Tuple[Tensor]:
        """
        sample points in the free space of the floor map to fill the map
        where each point cover a square area around it

        return
            points: (n,2) row,col indices of the points
            uncovered_mask: (h,w) binary mask, 1 is uncovered, 0 is covered
        """
        floor_map = self.floor_map.clone()
        points = []
        while True:
            # get regions and area and filter
            label_im, nb_labels = ndimage.label(
                floor_map.cpu().numpy().astype(np.uint8)
            )
            props = regionprops(label_im.astype(np.uint8))
            regions = [
                prop
                for prop in props
                if prop.axis_minor_length > self.min_size and prop.area > self.min_area
            ]

            if not len(regions):
                break

            # loop through area
            all_new_points = []
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                region_mask = floor_map[min_row:max_row, min_col:max_col]
                new_points, uncovered_mask = self._fill_map(region_mask)
                new_points[:, 0] += min_row
                new_points[:, 1] += min_col
                all_new_points.append(new_points)
                floor_map[min_row:max_row, min_col:max_col] = uncovered_mask
            all_new_points = torch.cat(all_new_points)
            if not len(all_new_points):
                break

            points.append(all_new_points)

        points = torch.cat(points)
        return points, floor_map


class DoorWindowExtractor:
    """
    extract door, window objects from the house
    """

    DOOR_CLASSES = ["door", "hole", "pocket"]
    WINDOW_CLASSES = ["window", "hole", "pocket"]

    def __init__(self, house):
        self.house = house

    def extract_doors(self):
        """
        # get door, hole objects

        # get center point of bottom rectangle of door, hole

        # get floor mesh of each room

        # for each hole, room find 2 nearest room and add edge between them
        """
        out = []
        # get room floor mesh
        room_floor_mesh = {}
        for r_name, room in self.house.rooms.items():
            r_f_mesh, r_nf_mesh = room.get_meshes()
            # get floor mesh
            floor_mesh = [x[1] for x in r_nf_mesh if "floor" in x[0]]
            if len(floor_mesh) == 0:
                continue
            floor_mesh = [
                o3dmesh_from_np(x["vertices"], x["faces"]) for x in floor_mesh
            ]
            floor_mesh = sum(floor_mesh, o3d.geometry.TriangleMesh())
            floor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(floor_mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(floor_mesh)
            room_floor_mesh[r_name] = scene

        for r_name, room in self.house.rooms.items():
            r_f_mesh, r_nf_mesh = room.get_meshes()

            # get door from pocket/door/hole
            doors = []
            for CLASS in self.DOOR_CLASSES:
                meshes = [x[1] for x in r_nf_mesh if CLASS in x[0]]
                meshes = [o3dmesh_from_np(x["vertices"], x["faces"]) for x in meshes]
                doors.extend(getattr(self, f"_get_door_from_{CLASS}")(meshes))

            # add room
            for door in doors:
                door.rooms.append(r_name)
                min_dist = np.inf
                best_room = None
                if len(door.bottom_vertices) == 0:
                    raise ValueError("door has no bottom vertices")
                bottom_center = door.bottom_vertices.mean(axis=0)
                for r_name2, room2 in room_floor_mesh.items():
                    if r_name == r_name2:
                        continue
                    query_point = o3d.core.Tensor(
                        [list(bottom_center)], dtype=o3d.core.Dtype.Float32
                    )
                    dist = room2.compute_distance(query_point)
                    if dist < min_dist:
                        min_dist = dist
                        best_room = r_name2
                if best_room is not None and min_dist < 0.5:
                    door.rooms.append(best_room)
            out.extend(doors)

            # TODO add door from other class
        return out

    def _get_door_from_pocket(
        self, pockets: List[o3d.geometry.TriangleMesh]
    ) -> List[Door]:
        """
        each door has 3 pocket objects, we ignore 2 pockets that has horizontal normal with all surfaces
        the door pocket has bottom part near zero while the widow does not

        args:
            pockets: list of pocket objects read from the house lengh N
        return list of door objects
        """
        outs = []
        for pocket in pockets:
            # check if pocket is window
            vertices = np.asarray(pocket.vertices)
            faces = np.asarray(pocket.triangles)
            if faces.shape[0] > 6:
                continue
            if vertices[:, 1].min() > 0.15:
                continue

            # check if the door pocket has horizontal normal with all surfaces
            pocket.compute_triangle_normals()
            normals = np.asarray(pocket.triangle_normals)
            vertical_normals = np.abs(normals[:, 1])
            if np.all(vertical_normals < 0.1):
                continue

            max_height = vertices[:, 1].max()
            min_height = vertices[:, 1].min()
            height = max_height - min_height
            if height < 0.1:
                bottom_vertices = vertices
            else:
                mid_height = (max_height + min_height) / 2
                bottom_vertices = vertices[vertices[:, 1] < mid_height]
            assert len(bottom_vertices) > 0, "door has no bottom vertices"
            outs.append(Door(bottom_vertices, height, []))
        return outs

    def _get_door_from_hole(self, holes: List[o3d.geometry.TriangleMesh]) -> List[Door]:
        """
        filter out hole for the door
        """
        outs = []

        for hole in holes:
            vertices = np.asarray(hole.vertices)
            if vertices.shape[0] < 8:
                continue
            if vertices[:, 1].min() > 0.15:
                continue

            max_height = vertices[:, 1].max()
            min_height = vertices[:, 1].min()
            height = max_height - min_height
            if height < 0.1:
                bottom_vertices = vertices
            else:
                mid_height = (max_height + min_height) / 2
                bottom_vertices = vertices[vertices[:, 1] < mid_height]
            assert len(bottom_vertices) > 0, "door has no bottom vertices"
            outs.append(Door(bottom_vertices, height, []))
        return outs

    def _get_door_from_door(self, doors: List[o3d.geometry.TriangleMesh]) -> List[Door]:
        return self._get_door_from_hole(doors)

    def extract_windows(self):
        """
        # get window, hole objects

        # get center point of bottom rectangle of window, hole

        # get floor mesh of each room

        # for each hole, room find 2 nearest room and add edge between them
        """
        out = []
        # get room floor mesh
        room_floor_mesh = {}
        for r_name, room in self.house.rooms.items():
            r_f_mesh, r_nf_mesh = room.get_meshes()
            # get floor mesh
            floor_mesh = [x[1] for x in r_nf_mesh if "floor" in x[0]]
            if len(floor_mesh) == 0:
                continue
            floor_mesh = [
                o3dmesh_from_np(x["vertices"], x["faces"]) for x in floor_mesh
            ]
            floor_mesh = sum(floor_mesh, o3d.geometry.TriangleMesh())
            floor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(floor_mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(floor_mesh)
            room_floor_mesh[r_name] = scene

        for r_name, room in self.house.rooms.items():
            r_f_mesh, r_nf_mesh = room.get_meshes()

            # get window from pocket/window/hole
            windows = []
            for CLASS in self.WINDOW_CLASSES:
                meshes = [x[1] for x in r_nf_mesh if CLASS in x[0]]
                meshes = [o3dmesh_from_np(x["vertices"], x["faces"]) for x in meshes]
                windows.extend(getattr(self, f"_get_window_from_{CLASS}")(meshes))

            # add room
            for window in windows:
                window.rooms.append(r_name)
                min_dist = np.inf
                best_room = None
                bottom_center = window.bottom_vertices.mean(axis=0)
                for r_name2, room2 in room_floor_mesh.items():
                    if r_name == r_name2:
                        continue
                    query_point = o3d.core.Tensor(
                        [list(bottom_center)], dtype=o3d.core.Dtype.Float32
                    )
                    dist = room2.compute_distance(query_point)
                    if dist < min_dist:
                        min_dist = dist
                        best_room = r_name2
                if best_room is not None and min_dist < 0.5:
                    window.rooms.append(best_room)
            out.extend(windows)

            # TODO add door from other class
        return out

    def _get_window_from_pocket(
        self, pockets: List[o3d.geometry.TriangleMesh]
    ) -> List[Window]:
        """
        each window has up to3 pocket objects, we ignore 2 pockets that has horizontal normal with all surfaces
        the window pocket has bottom part near zero while the widow does not

        args:
            pockets: list of pocket objects read from the house lengh N
        return list of window objects
        """
        outs = []
        for pocket in pockets:
            # check if pocket is window
            vertices = np.asarray(pocket.vertices)
            faces = np.asarray(pocket.triangles)
            if faces.shape[0] > 6:
                continue
            if vertices[:, 1].min() <= 0.15:
                continue

            # check if the window pocket has horizontal normal with all surfaces
            pocket.compute_triangle_normals()
            normals = np.asarray(pocket.triangle_normals)
            vertical_normals = np.abs(normals[:, 1])
            if np.all(vertical_normals < 0.1):
                continue

            max_height = vertices[:, 1].max()
            min_height = vertices[:, 1].min()
            height = max_height - min_height
            if height < 0.1:
                bottom_vertices = vertices
            else:
                mid_height = (max_height + min_height) / 2
                bottom_vertices = vertices[vertices[:, 1] < mid_height]
            outs.append(Window(bottom_vertices, height, []))
        return outs

    def _get_window_from_hole(
        self, holes: List[o3d.geometry.TriangleMesh]
    ) -> List[Window]:
        """
        filter out hole for the window
        """
        outs = []

        for hole in holes:
            vertices = np.asarray(hole.vertices)
            if vertices.shape[0] < 8:
                continue
            if vertices[:, 1].min() <= 0.15:
                continue

            max_height = vertices[:, 1].max()
            min_height = vertices[:, 1].min()
            height = max_height - min_height
            if height < 0.1:
                bottom_vertices = vertices
            else:
                mid_height = (max_height + min_height) / 2
                bottom_vertices = vertices[vertices[:, 1] < mid_height]
            outs.append(Window(bottom_vertices, height, []))
        return outs

    def _get_window_from_window(
        self, windows: List[o3d.geometry.TriangleMesh]
    ) -> List[Window]:
        return self._get_window_from_hole(windows)


@dataclass
class Door:
    bottom_vertices: np.ndarray
    height: float
    rooms: Optional[Tuple[str, str]] = None


@dataclass
class Window:
    bottom_vertices: np.ndarray
    height: float
    rooms: Optional[Tuple[str, str]] = None


def o3dmesh_from_np(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh



if __name__ == "__main__":
    import json
    args = parse_args()
    # scene_id = f"64e2e0ba-3769-479d-81d7-c870da620b07"
    # scene_id = f"644f3b6e-ad35-4254-92ea-626e3e8a65b1"
    # scene_id = f"v1_644f3b6e-ad35-4254-92ea-626e3e8a65b1"
    # args.front_folder = "/media/hieu/T7/datasets/3d-front/3D-FRONT"
    # args.front_folder = "/home/hieu/Downloads"
    # args.future_folder = "/media/hieu/T7/datasets/3d-front/3D-FUTURE-model"
    # args.future_bbox_folder = "/media/hieu/T7/3d-front/model_bbox__"
    # args.graph_folder = "/mnt/DATA/personal_projects/BlenderProc-3DFront/visualization/3d_vis/test_graph"
    # args.error_folder = "."
    
    os.makedirs(args.graph_folder, exist_ok=True)
    os.makedirs(args.error_folder, exist_ok=True)
    print(args)
    
    scene_ids = os.listdir(args.front_folder)
    scene_ids = sorted([x.split(".")[0] for x in scene_ids if x.endswith(".json")])
    scene_ids = scene_ids[args.start:args.end]
    scene_ids = scene_ids[args.offset::args.step]
    # scene_ids = sorted(json.load(open("/work/vig/hieu/3dfront_data/val_scenes_300_3000.json")))
    # scene_ids = [
    #     "64cce374-230b-4fe2-8240-69f81c8cfb33",
    #     "6a03badb-61ab-4e22-9564-de2fd2ed1938",
    #     "6ae5c274-8d8c-4832-878c-ea4f74082dfa",
    #     "6b56575f-a746-4062-8296-b566d5de7b60",
    #     "65aa5c84-5d84-4785-adf8-0000c91aa79e",
    #     "651c37ce-c0cd-47c6-843f-3aa192235e39",
    #     "64321004-c08d-4334-984b-4bdbed4895d1",
    #     "648b8740-c145-4233-84f1-27f596c387e7",
    #     "644dba87-0d65-4897-912c-38185791b3c2",
    #     "66f67e6f-2c22-473c-9997-0690a62661ec",
    #     "6be7ef9a-98d9-4d32-8b0c-c0d2b1ca3bbe",
    #     "6cc1d22f-47b3-4a3a-81b4-19575f2ec3cd",
    #     "6cd50522-54ed-4854-85b1-c0dc5f34abd8",
    #     "6d13bd56-8931-486a-8490-61752942a6de",
    # ]
    for scene_id in tqdm(scene_ids):
        out_path = os.path.join(args.graph_folder, f"{scene_id}.pkl")
        if os.path.exists(out_path):
            continue
        try:
            G = get_graph(scene_id, args)
            pickle.dump(G, open(out_path, "wb"))
        except Exception as e:
            print(f"Error: {scene_id}")
            print(e)
            with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
                f.write(str(e))
        # break
        
"""
local
python get_eval_graph.py \
--front_folder /media/hieu/T7/datasets/3d-front/3D-FRONT \
--future_folder /media/hieu/T7/datasets/3d-front/3D-FUTURE-model \
--future_bbox_folder /media/hieu/T7/3d-front/model_bbox__ \
--graph_folder /mnt/DATA/personal_projects/BlenderProc-3DFront/visualization/3d_vis/test_graph \
--error_folder /mnt/DATA/personal_projects/BlenderProc-3DFront/visualization/3d_vis/test_error \
--start 0 \
--end 7000 \
--offset 0 \
--step 1


neu

python visualization/3d_vis/get_eval_graph.py \
--front_folder /work/vig/Datasets/3D-Front/3D-FRONT \
--future_folder /work/vig/Datasets/3D-Front/3D-FUTURE-model \
--future_bbox_folder /work/vig/hieu/3dfront_data/model_bbox  \
--graph_folder /work/vig/hieu/3dfront_data/graph_pano \
--error_folder /work/vig/hieu/3dfront_data/graph_pano_error \
--start 0 \
--end 7000 \
--offset 0 \
--step 1

python visualization/3d_vis/get_eval_graph.py \
--front_folder /work/vig/Datasets/3D-Front/3D-FRONT \
--future_folder /work/vig/Datasets/3D-Front/3D-FUTURE-model \
--future_bbox_folder /work/vig/hieu/3dfront_data/model_bbox  \
--graph_folder /work/vig/hieu/3dfront_data/graph_pano_train \
--error_folder /work/vig/hieu/3dfront_data/graph_pano_train_error \
--start 0 \
--end 7000 \
--offset 0 \
--step 1 \
--variance 0.3
"""