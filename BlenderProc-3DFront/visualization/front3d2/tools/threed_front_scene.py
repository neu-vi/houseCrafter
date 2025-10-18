# Revised from
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

from collections import Counter
from dataclasses import dataclass
from functools import cached_property, reduce, lru_cache
import quaternion as qt
import json

import os
import numpy as np
from PIL import Image

import trimesh

from .common import BaseScene
from visualization.utils.tools import normalize
FRONT3D2BLENDER = np.array([
    [1.0, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

@dataclass
class Asset:
    """Contains the information for each 3D-FUTURE model."""

    super_category: str
    category: str
    style: str
    theme: str
    material: str

    @property
    def label(self):
        return self.category


class ModelInfo(object):
    """Contains all the information for all 3D-FUTURE models.

    Arguments
    ---------
    model_info_data: list of dictionaries containing the information
                     regarding the 3D-FUTURE models.
    """

    def __init__(self, model_info_data):
        self.model_info_data = model_info_data
        self._model_info = None
        # List to keep track of the different styles, themes
        self._styles = []
        self._themes = []
        self._categories = []
        self._super_categories = []
        self._materials = []

    @property
    def model_info(self):
        if self._model_info is None:
            self._model_info = {}
            # Create a dictionary of all models/assets in the dataset
            for m in self.model_info_data:
                # Keep track of the different styles
                if m["style"] not in self._styles:
                    self._styles.append(m["style"])
                # Keep track of the different themes
                if m["theme"] not in self._themes and m["theme"] is not None:
                    self._themes.append(m["theme"])
                # Keep track of the different super-categories
                if m["super-category"] not in self._super_categories:
                    self._super_categories.append(m["super-category"])
                # Keep track of the different categories
                if m["category"] not in self._categories:
                    self._categories.append(m["category"])
                # Keep track of the different categories
                if m["material"] not in self._materials:
                    self._materials.append(m["material"])

                self._model_info[m["model_id"]] = Asset(
                    (
                        m["super-category"].lower().replace(" / ", "/")
                        if m["super-category"]
                        else "others"
                    ),
                    (
                        m["category"].lower().replace(" / ", "/")
                        if m["category"]
                        else "others"
                    ),
                    m["style"] if m["style"] else "others",
                    m["theme"] if m["theme"] else "others",
                    m["material"] if m["material"] else "others",
                )

        return self._model_info

    @property
    def styles(self):
        return self._styles

    @property
    def themes(self):
        return self._themes

    @property
    def materials(self):
        return self._materials

    @property
    def categories(self):
        return set(
            [s.lower().replace(" / ", "/") if s else "others" for s in self._categories]
        )

    @property
    def super_categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._super_categories])

    @classmethod
    def from_file(cls, path_to_model_info):
        with open(path_to_model_info, "rb") as f:
            model_info = json.load(f)

        return cls(model_info)


class BaseThreedFutureModel(object):
    def __init__(self, model_uid, model_jid, position, rotation, scale):
        self.model_uid = model_uid
        self.model_jid = model_jid
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def _transform(self, vertices):
        # tranform from model coordinate system to world coordinate system of 3D-FRONT
        # then transform to blender coordinate system
        # 
        x, y, z, w = self.rotation
        m = qt.as_rotation_matrix(
            qt.quaternion(
                w,
                x,
                y,
                z,
            )
        )
        vertices = vertices @ m.T
        vertices += self.position
        
        vertices = vertices @ FRONT3D2BLENDER[:3,:3].T
        return vertices

    def get_world_orientation(self):
        x, y, z, w = self.rotation
        m = qt.as_rotation_matrix(
            qt.quaternion(
                w,
                x,
                y,
                z,
            )
        )
        m = FRONT3D2BLENDER[:3,:3] @ m
        q = qt.from_rotation_matrix(m)
        q = qt.as_float_array(q)
        return q

class ThreedFutureModel(BaseThreedFutureModel):
    def __init__(
        self,
        model_uid,
        model_jid,
        model_info,
        position,
        rotation,
        scale,
        path_to_models,
    ):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.model_info = model_info
        self.path_to_models = path_to_models
        self._label = None

    @property
    def raw_model_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "raw_model.obj")

    @property
    def texture_image_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "texture.png")

    @property
    def path_to_bbox_vertices(self):
        return os.path.join(self.path_to_models, self.model_jid, "bbox_vertices.npy")

    def raw_model(self):
        try:
            return trimesh.load(
                self.raw_model_path,
                process=False,
                force="mesh",
                skip_materials=True,
                skip_texture=True,
            )
        except:
            import pdb

            pdb.set_trace()
            print("Loading model failed", flush=True)
            print(self.raw_model_path, flush=True)
            raise

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        model = self.raw_model()
        faces = np.array(model.faces)
        vertices = self._transform(np.array(model.vertices)) + offset

        return trimesh.Trimesh(vertices, faces)

    def centroid(self, offset=[[0, 0, 0]]):
        return self.corners(offset).mean(axis=0)

    @cached_property
    def size(self):
        corners = self.corners()
        return np.array(
            [
                np.sqrt(np.sum((corners[4] - corners[0]) ** 2)),
                np.sqrt(np.sum((corners[2] - corners[0]) ** 2)),
                np.sqrt(np.sum((corners[1] - corners[0]) ** 2)),
            ]
        )

    def bottom_center(self, offset=[[0, 0, 0]]):
        centroid = self.centroid(offset)
        size = self.size
        return np.array([centroid[0], centroid[1] - size[1] / 2.0, centroid[2]])

    @cached_property
    def z_angle(self):
        """
        """
        raise NotImplementedError()

    @property
    def label(self):
        if self._label is None:
            self._label = self.model_info.label
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    def corners(self, offset=[[0, 0, 0]]):
        """Note x0<x1, y0<y1, z0<z1, then the 8 corners are concatenated by:
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
         [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]]"""
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except:
            bbox_vertices = np.array(self.raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        c = self._transform(bbox_vertices)
        return c + offset

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)

    def copy_from_other_model(self, other_model):
        model = ThreedFutureModel(
            model_uid=other_model.model_uid,
            model_jid=other_model.model_jid,
            model_info=other_model.model_info,
            position=self.position,
            rotation=self.rotation,
            scale=other_model.scale,
            path_to_models=self.path_to_models,
        )
        model.label = self.label
        return model


class ThreedFutureExtra(BaseThreedFutureModel):
    def __init__(
        self, model_uid, model_jid, xyz, faces, model_type, position, rotation, scale
    ):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.xyz = xyz
        self.faces = faces
        self.model_type = model_type

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        vertices = self._transform(np.array(self.xyz)) + offset
        faces = np.array(self.faces)
        return trimesh.Trimesh(vertices, faces)


class Room(BaseScene):
    def __init__(
        self, room_id, room_type, bboxes, extras, json_path, path_to_room_masks_dir=None
    ):
        super().__init__(room_id, room_type, bboxes)
        self.json_path = json_path
        self.extras = extras

        self.uid = "_".join([self.json_path, room_id])
        self.path_to_room_masks_dir = path_to_room_masks_dir
        if path_to_room_masks_dir is not None:
            self.path_to_room_mask = os.path.join(
                self.path_to_room_masks_dir, self.uid, "room_mask.png"
            )
        else:
            self.path_to_room_mask = None

    @property
    def floor(self):
        return [ei for ei in self.extras if ei.model_type == "Floor"][0]

    @property
    @lru_cache(maxsize=512)
    def layout_box(self):
        """
        get the layout box, which is in an array of [floor_center, x_vec, y_vec, z_vec] like below:
        |     |                      |     |
        |     |y_vec                 |     |
        |     |______________________|_____|
        |    /           / x_vec     |    /
        |  /          center -> z_vec|  /
        |/___________________________|/
        :return:
        """
        """get the bbox of room layout"""
        heading = self.room_orientation
        floor_coordinates = self.floor_plan[0]
        floor_height = np.mean(floor_coordinates[:, 1])
        # get room height
        room_top_height = np.max(
            [
                extra.xyz[:, 1].max()
                for extra in self.extras
                if extra.model_type == "WallInner"
            ]
        )
        room_height = room_top_height - floor_height
        # get room bbox
        z_projs = floor_coordinates.dot(heading[:, np.newaxis]) / (sum(heading**2))
        second_heading = np.cross([0, 1, 0], heading)
        x_projs = floor_coordinates.dot(second_heading[:, np.newaxis]) / (
            sum(second_heading**2)
        )
        max_z_proj, min_z_proj = np.max(z_projs), np.min(z_projs)
        max_x_proj, min_x_proj = np.max(x_projs), np.min(x_projs)
        z_coeff = (max_z_proj + min_z_proj) / 2.0
        x_coeff = (max_x_proj + min_x_proj) / 2.0
        floor_center = z_coeff * heading + x_coeff * second_heading
        z_vec = (max_z_proj - min_z_proj) / 2.0 * heading
        x_vec = (max_x_proj - min_x_proj) / 2.0 * second_heading
        y_vec = np.array([0, room_height, 0])
        return np.hstack([floor_center, x_vec, y_vec, z_vec])

    @property
    @lru_cache(maxsize=512)
    def bbox(self):
        corners = np.empty((0, 3))
        for f in self.bboxes:
            corners = np.vstack([corners, f.corners()])
        return np.min(corners, axis=0), np.max(corners, axis=0)

    @cached_property
    def bboxes_centroid(self):
        a, b = self.bbox
        return (a + b) / 2

    @property
    def furniture_in_room(self):
        return [f.label for f in self.bboxes]

    @property
    def room_orientation(self):
        """get the orientation of the room. (angle to z-axis)"""
        all_normals = []
        for extra in self.extras:
            if not extra.model_type == "WallInner":
                continue
            normals = [
                np.cross(
                    extra.xyz[face[1]] - extra.xyz[face[0]],
                    extra.xyz[face[2]] - extra.xyz[face[1]],
                )
                for face in extra.faces
            ]
            normals = [normal for normal in normals if np.linalg.norm(normal) > 0.0]
            all_normals.extend(normals)
        x = np.array(all_normals)[:, [2, 0]]  # z, x axes
        x = normalize(x)

        # move all normals into 1st quadrant
        x[x[:, 0] < 0] *= -1
        clockwise_rot90 = np.array([[0, 1], [-1, 0]])

        indices = x[:, 1] < 0
        x[indices] = x[indices].dot(clockwise_rot90)

        indices = x[:, 1] > x[:, 0]
        x[indices] = x[indices].dot(clockwise_rot90.T)

        indices = np.linalg.norm(x - [1.0, 0.0], axis=-1) < 1e-3
        x[indices] = np.array([1.0, 0.0])

        # get the dominated orientation
        values, counts = np.unique(x, axis=0, return_counts=True)
        primary_vec = values[np.argmax(counts)]
        room_normal = np.array([primary_vec[1], 0, primary_vec[0]])
        return room_normal

    @property
    def floor_plan(self):
        def cat_mesh(m1, m2):
            v1, f1 = m1
            v2, f2 = m2
            v = np.vstack([v1, v2])
            f = np.vstack([f1, f2 + len(v1)])
            return v, f

        # Compute the full floor plan
        vertices, faces = reduce(
            cat_mesh,
            ((ei.xyz, ei.faces) for ei in self.extras if ei.model_type == "Floor"),
        )
        return np.copy(vertices), np.copy(faces)

    @cached_property
    def floor_plan_bbox(self):
        vertices, faces = self.floor_plan
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    @cached_property
    def floor_plan_centroid(self):
        a, b = self.floor_plan_bbox
        return (a + b) / 2

    @cached_property
    def centroid(self):
        return self.floor_plan_centroid

    @property
    def count_furniture_in_room(self):
        return Counter(self.furniture_in_room)

    @property
    def room_mask(self):
        return self.room_mask_rotated(0)

    def room_mask_rotated(self, angle=0):
        # The angle is in rad
        im = Image.open(self.path_to_room_mask).convert("RGB")
        # Downsample the room_mask image by applying bilinear interpolation
        im = im.rotate(angle * 180 / np.pi, resample=Image.BICUBIC)

        return np.asarray(im).astype(np.float32) / np.float32(255)

    def category_counts(self, class_labels):
        """List of category counts in the room"""
        print(class_labels)
        category_counts = [0] * len(class_labels)

        for di in self.furniture_in_room:
            category_counts[class_labels.index(di)] += 1
        return category_counts

    def ordered_bboxes_with_centroid(self):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_labels(self, all_labels):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        int_labels = np.array([[f.int_label(all_labels)] for f in self.bboxes])
        ordering = np.lexsort(np.hstack([centroids, int_labels]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_frequencies(self, class_order):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        label_order = np.array([[class_order[f.label]] for f in self.bboxes])
        ordering = np.lexsort(np.hstack([centroids, label_order]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering[::-1]]

        return ordered_bboxes

    def augment_room(self, objects_dataset):
        bboxes = self.bboxes
        # Randomly pick an asset to be augmented
        bi = np.random.choice(self.bboxes)
        query_label = bi.label
        query_size = bi.size + np.random.normal(0, 0.02)
        # Retrieve the new asset based on the size of the picked asset
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )
        bi_retrieved = bi.copy_from_other_model(furniture)

        new_bboxes = [box for box in bboxes if not box == bi] + [bi_retrieved]

        return Room(
            room_id=self.room_id + "_augm",
            room_type=self.room_type,
            bboxes=new_bboxes,
            extras=self.extras,
            json_path=self.json_path,
            path_to_room_masks_dir=self.path_to_room_masks_dir,
        )
