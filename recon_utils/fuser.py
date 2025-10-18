import numpy as np
import open3d as o3d
import torch
import trimesh

class DepthFuser():
    def __init__(
            self,
            gt_path="", 
            fusion_resolution=0.04, 
            max_fusion_depth=3.0, 
            fuse_color=False
        ):
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth


class Open3DFuser(DepthFuser):
    """ 
    Wrapper class for the open3d fuser. 
    
    This wrapper does not support fusion of tensors with higher than batch 1.ß
    """
    def __init__(
            self, 
            gt_path="", 
            fusion_resolution=0.001, 
            max_fusion_depth=6, 
            fuse_color=False, 
            use_upsample_depth=False,
            front3d_base_matrix=None,
        ):
        super().__init__(
                    gt_path, 
                    fusion_resolution, 
                    max_fusion_depth,
                    fuse_color,
                )
        # print('Open3d fuser initialized')

        self.fuse_color = fuse_color
        self.use_upsample_depth = use_upsample_depth
        self.fusion_max_depth = max_fusion_depth

        voxel_size = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        if front3d_base_matrix is not None:
            self.FRONT3D_BASE_MATRIX = front3d_base_matrix
        else:
            self.FRONT3D_BASE_MATRIX = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )


    def fuse_frames(
            self, 
            depths_b1hw, 
            K_b44, 
            cam_T_world_b44, # is actually world to cam
            color_b3hw,
            size = None
        ):

        if size is None:
            width = depths_b1hw.shape[-1]
            height = depths_b1hw.shape[-2]
        else:
            width = size
            height = size

        if self.fuse_color:
            color_b3hw = torch.nn.functional.interpolate(
                                                    color_b3hw,
                                                    size=(height, width),
                                                )
            # color_b3hw = reverse_imagenet_normalize(color_b3hw)
            
        for batch_index in range(depths_b1hw.shape[0]):
            if self.fuse_color:
                image_i = color_b3hw[batch_index].permute(1,2,0)

                color_im = (image_i * 255).cpu().numpy().astype(
                                                            np.uint8
                                                        ).copy(order='C')
            else:
                # mesh will now be grey
                color_im = 0.7*torch.ones_like(
                                    depths_b1hw[batch_index]
                                ).squeeze().cpu().clone().numpy()
                color_im = np.repeat(
                                color_im[:, :, np.newaxis] * 255, 
                                3,
                                axis=2
                            ).astype(np.uint8)

            depth_pred = depths_b1hw[batch_index].squeeze().cpu().clone().numpy()

            depth_pred = o3d.geometry.Image(depth_pred)
            color_im = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                            color_im, 
                                            depth_pred, 
                                            depth_scale=1.0,
                                            depth_trunc=self.fusion_max_depth,
                                            convert_rgb_to_intensity=False,
                                        )
            cam_intr = K_b44[batch_index].cpu().clone().numpy()
            cam_T_world_44 = cam_T_world_b44[batch_index].cpu().clone().numpy()
            # print('cam_T_world_44:', cam_T_world_44.shape)

            cam_T_world_44[:3, :3] =  cam_T_world_44[:3, :3] @ self.FRONT3D_BASE_MATRIX
            
            self.volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    width=width, 
                    height=height, fx=cam_intr[0, 0], 
                    fy=cam_intr[1, 1],
                    cx=cam_intr[0, 2],
                    cy=cam_intr[1, 2]
                ),
                np.linalg.inv(cam_T_world_44),
            )

    def export_mesh(self, path, use_marching_cubes_mask=None):
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())
    
    def get_mesh(self, export_single_mesh=None, convert_to_trimesh=False):
        mesh = self.volume.extract_triangle_mesh()

        if convert_to_trimesh:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
        
        return mesh

