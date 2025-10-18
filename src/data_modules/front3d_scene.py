from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from json import JSONEncoder
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import quaternion as qt
import shapely
import trimesh
from shapely.geometry import Polygon

DEFAULT_POS = np.array([0.0, 0.0, 0.0])  # xyz
DEFAULT_ROT = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion xyzw
DEFAULT_SCALE = np.array([1.0, 1.0, 1.0])  # xyz


def load_scene(
    json_file, model_dir, furniture_bbox_dir, ignore_kitchen_bathroom=False
) -> House:
    with open(json_file, "r") as f:
        data = json.load(f)

    # load non-furniture
    non_furniture = {}
    for nf in data["mesh"]:
        nf_obj = NonFurniture.from_dict(nf)
        non_furniture[nf_obj.uid] = nf_obj

    # load furniture
    furniture = {}
    for f in data["furniture"]:
        if "valid" in f and f["valid"]:
            f_obj = Furniture.from_dict(f, model_dir, furniture_bbox_dir)
            furniture[f_obj.uid] = f_obj

    # make house
    house = House.from_dict(data)
    rooms = {}
    count = 0
    for r in data["scene"]["room"]:
        if ignore_kitchen_bathroom:
            if "kitchen" in r["type"].lower() or "bathroom" in r["type"].lower():
                continue
        room = Room.from_dict(r)
        rooms[room.instanceid] = room
        children = {}
        for c in r["children"]:
            instance = Instance.from_dict(c)

            if instance.ref in furniture:
                f = furniture[instance.ref]
                if f.used:
                    f = f.clone()
                instance.set_object(f)
            elif instance.ref in non_furniture:
                nf = non_furniture[instance.ref]
                if nf.used:
                    nf = nf.clone()
                instance.set_object(nf)
            else:
                count += 1

            if instance.object is not None:
                children[instance.instanceid] = instance
        room.children = children
    house.rooms = rooms
    return house


class House:
    def __init__(self, uid):
        self.uid = uid
        self.rooms: Dict[str, Room] = {}

    @classmethod
    def from_dict(cls, data_dict):
        house = cls(data_dict["uid"])
        assert check_pose(
            data_dict["scene"]
        ), f"Scene pose is not default, {data_dict['uid']}"
        # TODO may read mesh and room here
        return house

    def get_meshes(self) -> Tuple[List[Dict[str, np.ndarray]]]:
        """
        get meshes of all children in the house
        """
        f_meshes, nf_meshes = [], []
        for room in self.rooms.values():
            r_f_mesh, r_nf_mesh = room.get_meshes()
            f_meshes.extend(r_f_mesh)
            nf_meshes.extend(r_nf_mesh)
        return f_meshes, nf_meshes

    def get_ceil_floor_coords(self):
        """
        assume y is the up direction in world coordinate
        assume some edges of bbox are parallel to the floor
        get the min y of all instance in the house as the floor
        get the max y of all instance in the house as the ceil
        """
        floor = float("inf")
        ceil = -float("inf")
        for room in self.rooms.values():
            for instance in room.children.values():
                y_corners = instance.get_bbox().get_corners()[:, 1]
                floor = min(floor, y_corners.min())
                ceil = max(ceil, y_corners.max())
        return ceil, floor

    @property
    @lru_cache(maxsize=512)
    def floor_polygons(self) -> List[Polygon]:
        polygons = []
        for room in self.rooms.values():
            polygons.extend(room.floor_polygons)
        return polygons

    def bound(self):
        """
        return xyzxyz
        """
        bounds = []
        for room in self.rooms.values():
            bound = room.bound()
            if bound is not None:
                bounds.append(bound)
        bounds = np.stack(bounds)
        min_bounds = bounds.min(axis=0)
        max_bounds = bounds.max(axis=0)
        return np.concatenate([min_bounds[:3], max_bounds[-3:]])

    def sample_locations(
        self,
        dist=0.5,
        exclude_furniture=False,
        variance=0.2,
        furniture_margin=0.2,
        wall_margin=0.2,
        seed=0,
        height_range=[1.2, 1.7],
    ):
        points2d = []
        room_names = sorted(list(self.rooms.keys()))
        for i, name in enumerate(room_names):
            # if "secondbedroom" not in name.lower():
            #     continue
            room = self.rooms[name]
            points2d.append(
                room.get_locations_on_floor(
                    dist=dist,
                    exclude_furniture=exclude_furniture,
                    variance=variance,
                    wall_margin=wall_margin,
                    furniture_margin=furniture_margin,
                    seed=seed + i,
                )
            )
        rg = np.random.default_rng(seed)
        points2d = np.concatenate(points2d)
        heights = rg.uniform(height_range[0], height_range[1], points2d.shape[0])
        locations = np.stack([points2d[:, 0], heights, points2d[:, 1]], axis=1)
        return locations

    def get_furniture_2dbox(self, label_id_mapping, model_label) -> List[Dict]:
        """
        get 2d bounding boxes of all furniture in the house
        args:
            label_id_mapping: dict, mapping of label to id
            model_label: dict, mapping of model id to label
        return list id dict {
            bbox: np.array (4,2) (x,z),
            instanceid: str
            model_uid: str
            model_jid: str
            label: str
            label_id: int
        }
        """
        boxes = []
        for room in self.rooms.values():
            boxes.extend(room.get_furniture_2dbox())
        for box in boxes:
            box["label"] = model_label[box["model_jid"]]
            box["label_id"] = label_id_mapping[box["label"]]

        return boxes

    def save_furniture_2d_boxes(self, label_id_mapping, model_label, output_path):
        boxes = self.get_furniture_2dbox(label_id_mapping, model_label)
        ceil, floor = self.get_ceil_floor_coords()
        output = {"ceiling_height": ceil, "floor_height": floor, "boxes": boxes}
        json.dump(output, open(output_path, "w"), cls=NumpyArrayEncoder)

    @classmethod
    def sample_orientations(
        cls, locations, elevation_range=[-30, 30], seed=0, format="pose_matrix"
    ):
        """
        sample orientation for each location
        assume y is the up direction in world coordinate
        and the camera coordinate is blender camera coordinate
        (x right, y up, z backward)
        args:
            locations: np.array (n,3) x,y,z
            elevation_range: list, [min, max] in degree
            format: "pose_matrix" return 4x4 pose matrix
        """
        rg = np.random.default_rng(seed)
        headings = rg.uniform(0, 2 * np.pi, locations.shape[0])
        elevation_range = np.deg2rad(elevation_range)
        elevation = rg.uniform(*elevation_range, locations.shape[0])
        if format == "pose_matrix":
            out = np.zeros((locations.shape[0], 4, 4), dtype=np.float32)
            hq = qt.from_rotation_vector(headings.reshape(-1, 1) * np.array([0, 1, 0]))
            eq = qt.from_rotation_vector(elevation.reshape(-1, 1) * np.array([1, 0, 0]))
            rot_m = qt.as_rotation_matrix(hq * eq).astype(np.float32)
            out[:, :3, :3] = rot_m
            out[:, :3, 3] = locations
            out[:, 3, 3] = 1

        return out


class Room:
    def __init__(self, r_type, instanceid):
        self.type = r_type
        self.instanceid = instanceid
        self.children: Dict[str, Instance] = {}

    @classmethod
    def from_dict(cls, data_dict):
        room = cls(data_dict["type"], data_dict["instanceid"])
        assert check_pose(data_dict), f"Room pose is not default, {data_dict['type']}"
        # TODO may read children here
        return room

    def get_meshes(self) -> Tuple[List[Dict[str, np.ndarray]]]:
        """
        get meshes of all children in the room
        separate meshes of furniture and non-furniture

        """
        f_meshes, nf_meshes = [], []
        for instance in self.children.values():
            if instance.is_furniture():
                f_meshes.append(instance.mesh)
            else:
                nf_meshes.append((instance.object.type, instance.mesh))
        return f_meshes, nf_meshes

    @property
    @lru_cache(maxsize=512)
    def floor_polygons(self) -> List[Polygon]:
        """
        get the polygons of the floor of the room
        assume y is the up direction in world coordinate
        """
        polygons = []
        for instance in self.children.values():
            if instance.is_furniture():
                continue
            if "floor" in instance.object.type.lower():
                mesh = instance.mesh
                vertices = mesh["vertices"][:, [0, 2]]
                faces = mesh["faces"]
                poly = [Polygon(x) for x in vertices[faces]]
                polygons.extend(poly)
        return polygons

    def bound(self):
        """
        return xyzxyz
        """
        bounds = []
        for instance in self.children.values():
            if not instance.is_furniture():
                bounds.append(instance.get_bbox().bound())
        if not len(bounds):
            return None
        bounds = np.stack(bounds)
        min_bounds = bounds.min(axis=0)
        max_bounds = bounds.max(axis=0)
        return np.concatenate([min_bounds[:3], max_bounds[-3:]])

    def get_locations_on_floor(
        self,
        dist=0.5,
        exclude_furniture=False,
        variance=0.2,
        wall_margin=0.2,
        furniture_margin=0.2,
        seed=0,
    ):
        """
        assume y-axis is up direction return xz coords of points

        args:
            dist: distance between points on grid
            variance: whether to add gaussian noise to the grid,
                the variance is relative to the distance N(0, dist*variance)
            exclude_furniture: whether to exclude points that are inside 2d box of the furniture
        NOTE: the points may be inside the non-furniture mesh. The renderer will exclude them.
        """
        bounds = self.bound()
        if bounds is None:
            return np.empty((0, 2))
        xs, zs = np.meshgrid(
            np.arange(bounds[0] + wall_margin, bounds[3] - wall_margin, dist),
            np.arange(bounds[2] + wall_margin, bounds[5] - wall_margin, dist),
            indexing="ij",
        )
        xz = np.stack([xs.flatten(), zs.flatten()], axis=1)  # n,2

        if variance > 0:
            rg = np.random.default_rng(seed)
            xz += rg.normal(0, dist * variance, xz.shape)

        # get point actually in the house
        floor_polygons = self.floor_polygons
        if len(floor_polygons):
            points = shapely.points(xz)
            inliers = [f.contains(points) for f in floor_polygons]
            inliers = sum(inliers) > 0
            xz = xz[inliers]
        else:
            return np.empty((0, 2))

        # exclude points inside furniture
        if exclude_furniture:
            f_boxes = self.get_furniture_2dbox(furniture_margin)
            if len(f_boxes):
                f_boxes = [Polygon(x["bbox"]) for x in f_boxes]
                points = shapely.points(xz)
                inliers = [f.contains(points) for f in f_boxes]
                inliers = sum(inliers) > 0
                xz = xz[~inliers]
        return xz

    # NOTE subject to change
    def get_furniture_2dbox(self, margin=0.0) -> List[Dict]:
        """
        get 2d bounding boxes of all furniture in the room
        return list id dict {
            bbox: np.array (4,2) (x,z),
            instanceid: str
            model_uid: str
            model_jid: str
        }

        """
        boxes = []
        for instance in self.children.values():
            if instance.is_furniture():
                boxes.append(
                    {
                        "bbox": instance.get_bbox().get_2dbox(margin),
                        "instanceid": instance.instanceid,
                        "model_uid": instance.object.uid,
                        "model_jid": instance.object.jid,
                    }
                )
        return boxes


class Instance:
    """
    wrapper of 3d object (either furniture or non-furniture (wall, floor, ceiling,...))
    that store the object canonical mesh and pose, scale in the scene
    """

    def __init__(self, ref, pos, rot, scale, instanceid):
        self.ref = ref
        self.pos = np.array(pos, dtype=np.float32)
        if np.any(np.isnan(self.pos)):
            raise ValueError(f"Invalid position {pos}")
        self.rot = rot
        if np.any(np.isnan(self.rot)):
            raise ValueError(f"Invalid rotation {rot}")
        self.scale = np.array(scale, dtype=np.float32)
        if np.any(np.isnan(self.scale)):
            raise ValueError(f"Invalid scale {scale}")
        self.instanceid = instanceid
        self.object: Union[Furniture, NonFurniture, None] = None

    def is_furniture(self) -> bool:
        return isinstance(self.object, Furniture)

    @classmethod
    def from_dict(cls, data_dict):
        keys = ["ref", "pos", "rot", "scale", "instanceid"]
        kwargs = {k: data_dict[k] for k in keys}
        return cls(**kwargs)

    def set_object(self, obj: Union[Furniture, NonFurniture]):
        assert self.object is None, "Object already set"
        self.object = obj

    def get_bbox(self) -> BBox:
        if self.object is None:
            return None
        bbox = self.object.bbox
        bbox = bbox.transform(self.pose_matrix)
        bbox = bbox.scale(self.scale)
        return bbox

    @property
    @lru_cache(maxsize=512)
    def pose_matrix(self):
        """
        only include rotation and translation
        """
        out = np.eye(4)
        x, y, z, w = self.rot
        m = qt.as_rotation_matrix(
            qt.quaternion(
                w,
                x,
                y,
                z,
            )
        )
        out[:3, :3] = m
        out[:3, 3] = self.pos
        return out

    def _apply_transform(self, vertices, apply_scale=True):
        """
        vertices: N x 3
        """
        if apply_scale:
            vertices = vertices * self.scale
        pose = self.pose_matrix
        return vertices @ pose[:3, :3].T + pose[:3, 3]

    @property
    @lru_cache(maxsize=512)
    def mesh(self):
        """
        get mesh of instance in the scene (i.e in world coordinate system)
        {
            "vertices": np.array, N,3
            "faces": np.array, N,3
        }
        """
        assert self.object is not None
        mesh = self.object.mesh
        vertices = self._apply_transform(mesh["vertices"])
        return {"vertices": vertices, "faces": mesh["faces"]}


class Furniture:
    def __init__(
        self, uid, jid, sourceCategoryId, category, model_path, bbox_path=None
    ):
        self.uid = uid
        self.jid = jid
        self.sourceCategoryId = sourceCategoryId
        self.category = category
        self.model_path = model_path
        self.bbox_path = bbox_path
        self._mesh = None
        self.used = False

    @classmethod
    def from_dict(cls, data_dict, model_path, bbox_path=None):
        keys = ["uid", "jid", "sourceCategoryId", "category"]
        kwargs = {k: data_dict.get(k, None) for k in keys}
        return cls(**kwargs, model_path=model_path, bbox_path=bbox_path)

    def clone(self) -> Furniture:
        return Furniture(
            uid=self.uid,
            jid=self.jid,
            sourceCategoryId=self.sourceCategoryId,
            model_path=self.model_path,
            bbox_path=self.bbox_path,
        )

    @property
    @lru_cache(maxsize=512)
    def bbox(self):
        bbox = None
        precomputed = False
        if self.bbox_path is not None:
            path = os.path.join(self.bbox_path, f"{self.uid}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    bbox = BBox.from_dict(json.load(f))
                precomputed = True
        if bbox is None:
            mesh = self.mesh
            bbox = BBox.from_mesh(mesh)
        if not precomputed:
            if self.bbox_path is not None:
                path = os.path.join(
                    self.bbox_path, f"{self.uid.replace('/', '_')}.json"
                )
                with open(path, "w") as f:
                    json.dump(bbox.to_dict(), f)
        return bbox

    def _load_mesh(self):
        data = trimesh.load(
            os.path.join(self.model_path, self.jid, "raw_model.obj"),
            process=False,
            force="mesh",
            skip_materials=True,
            skip_texture=True,
        )
        faces = np.array(data.faces)
        vertices = np.array(data.vertices, dtype=np.float32)
        return {"vertices": vertices, "faces": faces}

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = self._load_mesh()
        return self._mesh


class NonFurniture:
    def __init__(self, uid, jid, mesh, o_type):
        self.uid = uid
        self.jid = jid
        self.type = o_type
        # dict of vertices and faces, both are (*,3) arrays
        self.mesh = mesh
        self.used = False

    @classmethod
    def from_dict(cls, data_dict):
        mesh = {
            "vertices": np.array(data_dict["xyz"], dtype=np.float32).reshape(-1, 3),
            "faces": np.array(data_dict["faces"], dtype=np.int64).reshape(-1, 3),
        }
        return cls(
            uid=data_dict["uid"],
            jid=data_dict["jid"],
            o_type=data_dict["type"].lower(),
            mesh=mesh,
        )

    def clone(self) -> NonFurniture:
        return NonFurniture(
            uid=self.uid,
            jid=self.jid,
            mesh=copy.deepcopy(self.mesh),
            o_type=self.type,
        )

    @property
    @lru_cache(maxsize=512)
    def bbox(self):
        return BBox.from_mesh(self.mesh)


class BBox:
    def __init__(self, center: np.ndarray, rotation: qt.quaternion, size: np.ndarray):
        self.center = center
        self.rotation = rotation
        self.size = size

    @property
    def _canonical_side_vector(self):
        """
        side vector is the vector from center to to the face of the box
        return (3,3) each row is one side vector
        """
        return np.array(
            [
                [self.size[0] / 2, 0, 0],
                [0, self.size[1] / 2, 0],
                [0, 0, self.size[2] / 2],
            ]
        )

    def get_corners(self):
        """
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
         [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]]
        return (8,3)
        """
        center, side_vector = self.get_center_side_vector()
        coef = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )
        return center + coef @ side_vector

    def get_center_side_vector(self, margin=0.0):
        """
        return bbox in the center and sidevector format
        center is the center of the bbox
        side vector is the vector from center to to the face of the box
        the three size vectors correspond to three axis of the box in the canonical axis
        (size_x/2, 0, 0), (0, size_y/2, 0), (0, 0, size_z/2)
        """
        r = qt.as_rotation_matrix(self.rotation)
        size_vector = (self._canonical_side_vector + margin * np.eye(3)) @ r.T
        return self.center.copy(), size_vector

    def transform(self, transform: np.ndarray) -> BBox:
        """
        apply rotation and translation to the bbox
        """
        center = transform[:3, :3] @ self.center
        rotation = qt.from_rotation_matrix(transform[:3, :3]) * self.rotation

        if transform.shape[0] == 4:
            center += transform[:3, 3]

        return BBox(center, rotation, self.size.copy())

    def scale(self, scale: np.ndarray) -> BBox:
        """
        apply scaling to the bbox
        NOTE: this only apply scale to the canonical axis of the box
            not the world axis the box is in
        """
        return BBox(self.center.copy(), self.rotation.copy(), self.size * scale)

    def to_dict(self):
        return {
            "center": self.center.tolist(),
            "rotation_wxyz": qt.as_float_array(self.rotation).tolist(),
            "size": self.size.tolist(),
        }

    @classmethod
    def from_dict(cls, data_dict):
        return cls(
            center=np.array(data_dict["center"]),
            rotation=qt.quaternion(data_dict["rotation_wxyz"]),
            size=np.array(data_dict["size"]),
        )

    @classmethod
    def from_mesh(cls, mesh: Dict[str, np.ndarray]):
        vertices = mesh["vertices"]
        lower_bound = vertices.min(axis=0)
        upper_bound = vertices.max(axis=0)
        size = upper_bound - lower_bound
        center = (upper_bound + lower_bound) / 2
        rotation = qt.from_rotation_matrix(np.eye(3))
        return BBox(center, rotation, size)

    def get_2dbox(self, margin=0.0):
        """
        assume the y direction of canonical box is the up direction in world coordinate
        and the rotation does not change the y-axis
        return 2D bounding box of the 3D box in the form of polygon
            [[x0, z0], [x0, z1], [x1, z1], [x1, z0],
        """
        center, side_vector = self.get_center_side_vector(margin)
        center_xz = center[[0, 2]]
        side_vector_xz = side_vector[:, [0, 2]][[0, 2]]
        coef = np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ],
            dtype=np.float32,
        )
        return center_xz + coef @ side_vector_xz

    def bound(self):
        """
        return xyzxyz
        """
        corners = self.get_corners()
        return np.concatenate([corners.min(axis=0), corners.max(axis=0)])


def check_pose(pose_dict):
    pos = np.allclose(pose_dict["pos"], DEFAULT_POS)
    rot = np.allclose(pose_dict["rot"], DEFAULT_ROT)
    scale = np.allclose(pose_dict["scale"], DEFAULT_SCALE)
    return pos and rot and scale


def load_labels(label_id_mapping_file, model_info_file):
    label_id_mapping = pd.read_csv(label_id_mapping_file)
    label_id2name = label_id_mapping.set_index("id").to_dict()["name"]

    label_id_mapping.set_index("name", inplace=True)
    label_id_mapping = label_id_mapping.to_dict()["id"]

    model_infor = json.load(open(model_info_file))
    model_id_to_label = {
        m["model_id"]: (
            m["category"].lower().replace(" / ", "/") if m["category"] else "others"
        )
        for m in model_infor
    }
    return label_id_mapping, model_id_to_label, label_id2name


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
