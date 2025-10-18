import zlib

import cv2
import lmdb
import numpy as np
import open3d as o3d
import seaborn as sns
from einops import repeat
from front3d_scene import BBox, House


class Front3DViz:
    def __init__(
        self,
        house: House = None,
        pointcloud=None,
        cam_poses=None,
        locations=None,
        image_path=None,
        layout_pcd_path=None,
        label_id2name=None,
    ):
        self.house = house
        self.pointcloud = pointcloud
        self.cam_poses = cam_poses
        self.locations = locations
        self.image_path = image_path
        self.layout_pcd_path = layout_pcd_path
        self.label_id2name = label_id2name
        self._non_furniture_mesh = None
        self._furniture_mesh = None

    def show(self, keys, ceiling=False, floor=True, class_id=None, c2w=None):
        """
        keys: non_furniture, furniture, 2dlayout, pointcloud, camera_poses, 3dbox, floor
        """
        print("showing...")

        showing_objs = []
        if "non_furniture" in keys:
            showing_objs.extend(self.get_non_furniture(ceiling, floor))

        if "furniture" in keys:
            showing_objs.extend(self.get_furniture())

        if "2dlayout" in keys:
            showing_objs.extend(self.get_2dlayout())

        if "3dbox" in keys:
            showing_objs.extend(self.get_3dbox())

        if "floor" in keys:
            showing_objs.extend(self.get_floor())

        self.set_mesh_color(showing_objs)

        if "camera_poses" in keys:
            showing_objs.extend(self.get_camera_poses(c2w=c2w))

        if "locations" in keys:
            showing_objs.extend(self.get_locations())

        if "pointcloud" in keys:
            showing_objs.extend(self.get_pointcloud())

        if "layout_pcd" in keys:
            showing_objs.extend(self.get_layout_pcd(class_id=class_id))

        o3d.visualization.draw_geometries(showing_objs)

    def get_furniture(self):
        if self._furniture_mesh is None:
            f_mesh, nf_mesh = self.house.get_meshes()
            self._furniture_mesh = f_mesh
            self._non_furniture_mesh = nf_mesh
        else:
            f_mesh = self._furniture_mesh

        o3d_meshes = []
        for mesh in f_mesh:
            o3d_mesh = self._make_open3d_mesh(mesh)
            o3d_meshes.append(o3d_mesh)
        return o3d_meshes

    def get_non_furniture(self, ceiling=False, floor=False):
        if self._non_furniture_mesh is None:
            f_mesh, nf_mesh = self.house.get_meshes()
            self._furniture_mesh = f_mesh
            self._non_furniture_mesh = nf_mesh
        else:
            nf_mesh = self._non_furniture_mesh

        o3d_meshes = []
        for name, mesh in nf_mesh:
            if not ceiling and "ceil" in name:
                continue
            if not floor and "floor" in name:
                continue
            o3d_mesh = self._make_open3d_mesh(mesh)
            o3d_meshes.append(o3d_mesh)
        return o3d_meshes

    def get_floor(self):
        if self._non_furniture_mesh is None:
            f_mesh, nf_mesh = self.house.get_meshes()
            self._furniture_mesh = f_mesh
            self._non_furniture_mesh = nf_mesh
        else:
            nf_mesh = self._non_furniture_mesh

        o3d_meshes = []
        for name, mesh in nf_mesh:
            if not "floor" in name.lower():
                continue
            o3d_mesh = self._make_open3d_mesh(mesh)
            o3d_meshes.append(o3d_mesh)
        return o3d_meshes

    def get_3dbox(self):
        o3d_boxes = []
        for room in self.house.rooms.values():
            for instance in room.children.values():
                if instance.is_furniture():
                    o3d_boxes.append(self._make_open3d_3dbox(instance.get_bbox()))
        return o3d_boxes

    def get_2dlayout(self):
        o3d_boxes = []
        ceil, floor = self.house.get_ceil_floor_coords()
        for room in self.house.rooms.values():
            for instance in room.children.values():
                if instance.is_furniture():
                    bbox = instance.get_bbox().get_2dbox()
                    # add floor coor to bbox
                    bbox = np.stack(
                        [bbox[:, 0], np.ones(4) * floor, bbox[:, 1]], axis=1
                    )
                    o3d_boxes.append(self._make_open3d_2d_box(bbox))
        return o3d_boxes

    def get_locations(self):
        o3d_spheres = []
        for point in self.locations:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=5)
            sphere.translate(point)
            o3d_spheres.append(sphere)
        return o3d_spheres

    def get_pointcloud(self):
        env = lmdb.open(
            self.image_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        pcds = []
        colors = []
        with env.begin() as txn:
            exist_keys = list(txn.cursor().iternext(values=False))
            # exist_keys = [key.decode().split('_')[0] for key in exist_keys]
            exist_keys = [key.decode() for key in exist_keys]
            img_keys = [key for key in exist_keys if "rgb" in key]
            for i, img_key in enumerate(img_keys):

                img = txn.get(img_key.encode("ascii"))
                img = np.frombuffer(img, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                pose_key = img_key[:-4] + "_pose"
                pose = txn.get(pose_key.encode("ascii"))
                pose = np.frombuffer(zlib.decompress(pose), dtype=np.float32).reshape(
                    4, 4
                )

                depth_key = img_key[:-4] + "_depth"
                depth = zlib.decompress(txn.get(depth_key.encode("ascii")))
                depth = np.frombuffer(depth, dtype=np.uint16).reshape(512, 512)
                depth = depth.astype(np.float32) / 1000.0

                points, color = PointCloud.get_point_cloud(depth, img, pose)
                pcds.append(points)
                colors.append(color)
        pcds_ = np.concatenate(pcds, axis=0)
        colors_ = np.concatenate(colors, axis=0)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcds_)
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors_ / 255)
        return [o3d_pcd]

    def get_layout_pcd(self, class_id=None):
        """
        class_id: if given only show point cloud of that class
        """
        env = lmdb.open(
            self.layout_pcd_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        pcds = []
        labels = []
        with env.begin() as txn:
            exist_keys = list(txn.cursor().iternext(values=False))
            # exist_keys = [key.decode().split('_')[0] for key in exist_keys]
            exist_keys = [key.decode() for key in exist_keys]
            img_keys = [key for key in exist_keys if not "pose" in key]
            for i, img_key in enumerate(img_keys):
                pose_key = img_key + "_pose"
                pose = txn.get(pose_key.encode("ascii"))
                pose = np.frombuffer(zlib.decompress(pose), dtype=np.float32).reshape(
                    4, 4
                )

                depth_key = img_key
                depth = zlib.decompress(txn.get(depth_key.encode("ascii")))
                depth = np.frombuffer(depth, dtype=np.uint16).reshape(-1, 2, 512, 512)
                depth_cls = depth[:, 0]
                depth = depth[:, 1]
                depth = depth.astype(np.float32) / 1000.0

                points, label = PointCloud.get_point_clouds(
                    depth[:], depth_cls[:], pose
                )
                pcds.append(points)
                labels.append(label)
        pcds_ = np.concatenate(pcds, axis=0)
        labels_ = np.concatenate(labels, axis=0)
        label_set = np.unique(labels_)
        label_set = {k: v for k, v in self.label_id2name.items() if k in label_set}
        if class_id is not None and class_id in label_set:
            mask = labels_ == class_id
            pcds_ = pcds_[mask]
        print(label_set)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcds_)
        return [o3d_pcd]

    def get_camera_poses(self, c2w=None):
        """
        o3d camera coordinate is x right, y up, z backward
        """
        if c2w is None:
            c2w = self.cam_poses.astype(np.float64)
            c2w[:, 3, 3] = 1.0
        w2c = np.linalg.inv(c2w)
        K = PointCloud.K.astype(np.float64)
        o3d_lines = []
        for p in w2c:
            lines = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=512,
                view_height_px=512,
                intrinsic=K,
                extrinsic=p,
                scale=0.05,
            )
            o3d_lines.append(lines)
        return o3d_lines

    @classmethod
    def _make_open3d_mesh(cls, mesh):
        vertices = o3d.utility.Vector3dVector(mesh["vertices"])
        triangles = o3d.utility.Vector3iVector(mesh["faces"])
        o3d_mesh = o3d.geometry.TriangleMesh(vertices, triangles)
        return o3d_mesh

    @classmethod
    def _make_open3d_3dbox(cls, bbox: BBox):
        corners = bbox.get_corners()
        lines = [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        return line_set

    @classmethod
    def _make_open3d_2d_box(cls, bbox):
        """
        bbox: [p1,p2,p3,p4] (4,3)
        """
        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        return line_set

    def set_mesh_color(self, geometries):
        meshes = [x for x in geometries if isinstance(x, o3d.geometry.TriangleMesh)]
        colors = np.array(sns.color_palette("hls", len(meshes)))
        for color, mesh in zip(colors, meshes):
            mesh.paint_uniform_color(color)


class PointCloud:
    K = np.array([[256.0, 0, 256], [0, 256, 256], [0, 0, 1]])
    BASE = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])

    @classmethod
    def get_point_cloud(cls, depth, rgb, P, num_sample=0):
        K = cls.K
        H, W = depth.shape
        ys, xs = np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32)
        ys = (ys - K[1, 2]) / K[1, 1]
        xs = (xs - K[0, 2]) / K[0, 0]
        ys, xs = np.meshgrid(ys, xs, indexing="ij")
        points = np.stack([xs * depth, ys * depth, depth], axis=-1)
        points = points[depth > 0.09]
        colors = rgb[depth > 0.09]
        if num_sample > 0:
            num_sample = min(num_sample, len(points))
            indices = np.random.choice(len(points), num_sample, replace=False)
            points = points[indices]
            colors = colors[indices]

        points = points @ cls.BASE.T @ P[:3, :3].T + P[:3, 3]
        return points, colors

    @classmethod
    def get_point_clouds(cls, depth, rgb, P, num_sample=10000):
        K = cls.K
        n, H, W = depth.shape
        ys, xs = np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32)
        ys = (ys - K[1, 2]) / K[1, 1]
        xs = (xs - K[0, 2]) / K[0, 0]
        ys, xs = np.meshgrid(ys, xs, indexing="ij")
        ys = repeat(ys, "h w -> n h w", n=n)
        xs = repeat(xs, "h w -> n h w", n=n)
        points = np.stack([xs * depth, ys * depth, depth], axis=-1)
        points = points[depth > 0.09]
        colors = rgb[depth > 0.09]
        if num_sample > 0:
            num_sample = min(num_sample, len(points))
            indices = np.random.choice(len(points), num_sample, replace=False)
            points = points[indices]
            colors = colors[indices]

        points = points @ cls.BASE.T @ P[:3, :3].T + P[:3, 3]
        return points, colors
