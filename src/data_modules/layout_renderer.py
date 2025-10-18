import json
import math

import pandas as pd
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene


class TorchLayoutRenderer:
    # transform from pytorch3d to matterport coordinate
    # torch: x-left, y-up, z-front
    # matterport: x-right, y-front, z-up
    # 3dfront/blender/opengl: x-right, y-up, z-back
    # transform camera from torch3d to 3dfront
    BASE_POSE = torch.tensor(
        [
            [-1.0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(
        self, device, image_size, znear=0.05, zfar=30.0, fov=90, filter_mesh_class=False
    ):
        self.device = device
        self.znear = znear
        self.zfar = zfar
        self.fov = fov
        self.image_size = image_size
        cameras = FoVPerspectiveCameras(device=device)
        self.raster_settings = RasterizationSettings(
            image_size=image_size, cull_backfaces=True
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )
        # self.shader = HardPhongShader(device=device, cameras=cameras)
        self.base_pose = self.BASE_POSE.to(device)

        self.house_mesh = None
        self.layout_mesh = None
        self.filter_mesh_class = filter_mesh_class

    def set_house_mesh(self, house_path, label_id_mapping_file):
        """
        read house mesh data
        """
        if self.filter_mesh_class:
            self.set_house_mesh_with_filtering(house_path, label_id_mapping_file)
            return
        data = json.load(open(house_path, "r"))
        label_mapping = load_label_mapping(label_id_mapping_file)
        mesh_objects = []
        labels = [0]
        for mesh_data in data["mesh"]:
            obj_name = mesh_data["type"].strip()
            if obj_name == "":
                obj_name = "void"
            # extract the vertices from the mesh_data
            vertices = torch.tensor(
                [float(ele) for ele in mesh_data["xyz"]], dtype=torch.float32
            ).reshape(-1, 3)
            # extract the faces from the mesh_data
            faces = torch.tensor(mesh_data["faces"], dtype=torch.int64).reshape(-1, 3)
            mesh = Meshes(verts=[vertices], faces=[faces])
            mesh_objects.append(mesh)
            labels.extend([label_mapping[obj_name.lower()]] * len(faces))

        self.house_mesh = join_meshes_as_scene(mesh_objects).to(self.device)
        self.house_labels = torch.tensor(labels, device=self.device, dtype=torch.long)

    def set_house_mesh_with_filtering(self, house_path, label_id_mapping_file):
        WALL_CLASSES = ["wall", "front", "back", "hole", "window", "door", "pocket"]

        data = json.load(open(house_path, "r"))
        label_mapping = load_label_mapping(label_id_mapping_file)
        mesh_objects = []
        labels = [0]
        for mesh_data in data["mesh"]:
            obj_name = mesh_data["type"].strip().lower()
            keep = True
            if any([ele in obj_name for ele in WALL_CLASSES]):
                obj_name = "wallinner"
            elif "ceiling" in obj_name:
                obj_name = "ceiling"
            elif "floor" in obj_name:
                obj_name = "floor"
            else:
                keep = False
            if not keep:
                continue
            # extract the vertices from the mesh_data
            vertices = torch.tensor(
                [float(ele) for ele in mesh_data["xyz"]], dtype=torch.float32
            ).reshape(-1, 3)
            # extract the faces from the mesh_data
            faces = torch.tensor(mesh_data["faces"], dtype=torch.int64).reshape(-1, 3)
            mesh = Meshes(verts=[vertices], faces=[faces])
            mesh_objects.append(mesh)
            labels.extend([label_mapping[obj_name.lower()]] * len(faces))

        self.house_mesh = join_meshes_as_scene(mesh_objects).to(self.device)
        self.house_labels = torch.tensor(labels, device=self.device, dtype=torch.long)

    def set_layout_mesh(self, layout_path, reverse=False, skip_ids=set()):
        furniture_layouts = json.load(open(layout_path))
        floor_height = furniture_layouts["floor_height"]
        ceiling_height = furniture_layouts["ceiling_height"]

        mesh_objects = []
        labels = []
        for box in furniture_layouts["boxes"]:
            if box["label_id"] in skip_ids:
                print(box)
                continue
            labels.append(box["label_id"])

            # make mesh for the box
            # while the box only have 4 corners, the following code works for any polygon
            # we assume the up direction is y-axis and the points follows the positive rotation of the y-axis
            vertices_xz = torch.tensor(box["bbox"])  # n,2
            n = vertices_xz.shape[0]
            floor_vertices = torch.stack(
                [vertices_xz[:, 0], torch.ones(n) * floor_height, vertices_xz[:, 1]],
                dim=1,
            )
            ceiling_vertices = torch.stack(
                [vertices_xz[:, 0], torch.ones(n) * ceiling_height, vertices_xz[:, 1]],
                dim=1,
            )
            vertices = torch.concatenate([ceiling_vertices, floor_vertices], dim=0)
            faces = []
            for i in range(n):
                faces.append([i, i + n, (i + 1) % n])
                faces.append([(i + 1) % n, i + n, (i + 1) % n + n])
            faces = torch.tensor(faces, dtype=torch.long)
            if reverse:
                faces = faces[:, [0, 2, 1]]
            mesh = Meshes(verts=[vertices], faces=[faces])
            mesh_objects.append(mesh)
        self.layout_mesh = join_meshes_as_batch(mesh_objects).to(self.device)
        self.layout_labels = torch.tensor(labels, device=self.device, dtype=torch.long)

    def render_layout(self, pose, batch_size=16):
        """
        args:
            pose: tensor (4,4) c2w in front3d convention, x right, y up, z forward

        return
            depth: (n,h,w) depth image (in meter)
            cls: (n,h,w) class image
            NOTE: the first channel is wall, the rest is object
        """
        assert self.layout_mesh is not None
        assert self.house_mesh is not None
        cameras = self._make_cameras(pose)

        wall_depth, wall_label = self._render(
            cameras, self.house_mesh.to(self.device), self.house_labels
        )
        not_avail_wall_depth = wall_depth < self.znear

        # render object depth
        depth_per_view = []
        label_per_view = []
        num_batch = math.ceil(len(self.layout_mesh) / batch_size)
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            batch_mesh = self.layout_mesh[start:end]
            batch_label = self.layout_labels[start:end]
            depth, _ = self._render(cameras, batch_mesh)
            has_obj = depth.amax(dim=(1, 2)) > self.znear
            depth = depth[has_obj]
            if depth.numel() == 0:
                continue
            keep_mask = wall_depth > depth
            ## do not remove where the wall_depth is not available
            keep_mask = torch.logical_or(keep_mask, not_avail_wall_depth)
            depth *= keep_mask

            batch_label = batch_label[has_obj.cpu()]
            depth_per_view.append(depth)
            label_per_view.append(batch_label)
        return self._post_process(
            wall_depth, wall_label, depth_per_view, label_per_view
        )

    @torch.no_grad()
    def _render(
        self, cameras: FoVPerspectiveCameras, mesh: Meshes, faceid2classid=None
    ):
        """
        run a rendering call with given cameras and mesh
        if faceid2classid is given, return class id for each pixel as well

        args:
            faceid2classid: (F+1,) longtensor
                if given, the mesh should have bs 1, number of faces is F
                and the faceid2classid give the class id for each face
                faceid2classid[0] is the class id for background

        return
            depth n,h,w,
            cls(optional) n,h,w
        """
        if faceid2classid is not None:
            assert mesh._F == faceid2classid.numel() - 1
            assert mesh._N == 1
        self.raster_settings.max_faces_per_bin = int(max(10000, mesh._F / 2))
        fragments = self.rasterizer(
            mesh, cameras=cameras, raster_settings=self.raster_settings
        )
        depths = fragments.zbuf.squeeze(-1)
        depths = depths * (depths > self.znear) * (depths < self.zfar)

        class_ids = None
        if faceid2classid is not None:
            class_ids = faceid2classid[fragments.pix_to_face.squeeze(-1) + 1]
        return depths, class_ids

    def _make_cameras(self, P):
        """
        P: c2w 4,4 in 3dfront convention
        """
        if len(P.shape) == 2:
            P = P.unsqueeze(0)
        P = P @ self.BASE_POSE
        P = torch.linalg.inv(P)  # w2c
        R = P[:, :3, :3].transpose(
            1, 2
        )  # transpose since torch use left multiplication
        T = P[:, :3, 3]
        return FoVPerspectiveCameras(
            znear=self.znear,
            zfar=self.zfar,
            fov=self.fov,
            R=R,
            T=T,
            device=self.device,
        )

    def _post_process(self, wall_depth, wall_label, depth_per_view, label_per_view):
        """
        wall_depth 1,h,w
        wall_label 1,h,w
        depth_per_view: list of depth n,h,w
        label_per_view: list of label n,

        return
            depth n,h,w,
            label n,h,w
        """
        if not len(depth_per_view):
            return wall_depth, wall_label

        depth_per_view = torch.cat(depth_per_view, dim=0)
        label_per_view = torch.cat(label_per_view, dim=0).view(-1, 1, 1)
        label_per_view = label_per_view * (depth_per_view > self.znear)
        n_max = (label_per_view > 0).sum(dim=0).max()
        if n_max == 0:
            return wall_depth, wall_label

        # sort by label
        label_per_view, idx = label_per_view.sort(dim=0, descending=True)
        idx = idx[:n_max]
        label_per_view = label_per_view[:n_max]
        depth_per_view = torch.take_along_dim(depth_per_view, idx, dim=0)

        out_depth = torch.cat([wall_depth, depth_per_view], dim=0)
        out_label = torch.cat([wall_label, label_per_view], dim=0)
        return out_depth, out_label


def load_label_mapping(label_id_mapping_file):  # , model_info_file):
    label_id_mapping = pd.read_csv(label_id_mapping_file)
    # label_id2name = label_id_mapping.set_index("id").to_dict()["name"]

    label_id_mapping.set_index("name", inplace=True)
    label_id_mapping = label_id_mapping.to_dict()["id"]

    # model_infor = json.load(open(model_info_file))
    # model_id_to_label = {
    #     m["model_id"]: (
    #         m["category"].lower().replace(" / ", "/") if m["category"] else "others"
    #     )
    #     for m in model_infor
    # }
    return label_id_mapping  # , model_id_to_label, label_id2name
