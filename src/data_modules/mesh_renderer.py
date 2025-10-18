from typing import Union

import numpy as np
import torch
import torch.nn as nn

try:
    from open3d.cuda.pybind.geometry import TriangleMesh
except ImportError:
    print("Open3D CUDA is not available")
    TriangleMesh = None
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    Textures,
)
from pytorch3d.renderer.blending import BlendParams, hard_rgb_blend
from pytorch3d.structures import Meshes


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


class TorchMeshRenderer:
    # transform from pytorch3d to matterport coordinate
    # torch: x-left, y-up, z-front
    # matterport: x-right, y-front, z-up
    # 3dfront/blender/opengl: x-right, y-up, z-back
    ZNEAR = 0.05
    ZFAR = 30.0
    # transform camera from torch3d to 3dfront
    BASE_POSE = torch.tensor(
        [
            [-1.0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, device, image_size, fov=90) -> None:
        self.device = device
        self.fov = fov
        self.image_size = image_size
        self.device = device
        cameras = FoVPerspectiveCameras(device=device)
        self.raster_settings = RasterizationSettings(
            image_size=image_size, cull_backfaces=True
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )
        # self.shader = HardPhongShader(device=device, cameras=cameras)
        self.shader = SimpleShader()
        self.base_pose = self.BASE_POSE.to(device)

    def _make_cameras(self, P):
        """
        P: c2w N,4,4 in 3dfront convention
        """
        P = P @ self.BASE_POSE
        P = torch.linalg.inv(P)  # w2c
        R = P[:, :3, :3].transpose(
            1, 2
        )  # transpose since torch use left multiplication
        T = P[:, :3, 3]
        return FoVPerspectiveCameras(
            znear=self.ZNEAR,
            zfar=self.ZFAR,
            fov=self.fov,
            R=R,
            T=T,
            device=self.device,
        )

    @torch.no_grad()
    def render(
        self, mesh: Union[Meshes, TriangleMesh], P: torch.Tensor
    ) -> torch.Tensor:
        """
        return depth n,h,w
            images n h w 3 (rgb in range [0,1])
        """
        if not isinstance(mesh, Meshes):
            mesh = self.o3d_mesh_to_torch(mesh, self.device)
        cameras = self._make_cameras(P)
        self.raster_settings.max_faces_per_bin = int(max(10000, mesh._F / 2))
        fragments = self.rasterizer(
            mesh, cameras=cameras, raster_settings=self.raster_settings
        )
        depths = fragments.zbuf.squeeze(-1)
        depths = depths * (depths > self.ZNEAR) * (depths < self.ZFAR)
        images = self.shader(
            fragments, mesh, cameras=cameras, raster_settings=self.raster_settings
        )
        return images[..., :3], depths

    @classmethod
    def o3d_mesh_to_torch(cls, mesh: TriangleMesh, device):
        vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
        faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.float32)
        colors = torch.tensor(np.asarray(mesh.vertex_colors), dtype=torch.float32)
        tex = Textures(verts_rgb=colors.unsqueeze(0).to(device))
        torch_mesh = Meshes(
            verts=[vertices.to(device)], faces=[faces.to(device)], textures=tex
        )
        return torch_mesh
