import numpy as np
import open3d as o3d
import torch
from torch import Tensor
import trimesh
import matplotlib.pyplot as plt
import os
from fuser import Open3DFuser
import lmdb
import cv2
import math
import zlib
import json
import shutil
import argparse
from denoise import denoise_mesh_by_connectedComponents

front3d_base_matrix = np.array(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
)

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_coord(H, W, K, pose, depth):
  R = torch.Tensor(pose)
  rays_o, rays_d = get_rays(H, W, K, R)
  depth = torch.Tensor(depth.astype(np.float32))
  depth = depth
  coord = rays_o + rays_d* depth.reshape((H,W,1))
  return coord

def load_frame_for_pcd(frame_id, img_path, resolution = 512):
    skip = False
    color_raw_path = os.path.join(img_path, 'colors', frame_id + '.png')
    depth_raw_path = os.path.join(img_path, 'depth', frame_id + '_rgb.npy')
    pose_raw_path = os.path.join(img_path, 'cam_Ts', frame_id + '.npy')
    color_raw_A = plt.imread(color_raw_path)
    depth_raw_A = np.load(depth_raw_path)
    depth_raw_A = depth_raw_A.astype(np.float32)*0.001
    pose_raw_A = (np.load(pose_raw_path))# c2w
    pose_raw_A[3][3] = 1.0
    # if size is not equal to resolution, resize
    if depth_raw_A.shape[0] != resolution:
        depth_raw_A = cv2.resize(depth_raw_A, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    if color_raw_A.shape[0] != resolution:
        color_raw_A = cv2.resize(color_raw_A, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    
    return depth_raw_A, pose_raw_A, color_raw_A, skip

def load_frame(frame_id, img_path, resolution = 512):
    skip = False
    color_raw_path = os.path.join(img_path, 'colors', frame_id + '.png')
    depth_raw_path = os.path.join(img_path, 'depth', frame_id + '_rgb.npy')
    pose_raw_path = os.path.join(img_path, 'cam_Ts', frame_id + '.npy')
    color_raw_A = plt.imread(color_raw_path)
    depth_raw_A = np.load(depth_raw_path)
    depth_raw_A = depth_raw_A.astype(np.float32)*0.001
    pose_raw_A = (np.load(pose_raw_path))# c2w
    pose_raw_A[3][3] = 1.0

    color_raw_A = color_raw_A[:,:,:3].transpose(2, 0, 1)
    skip = False
    color_A = color_raw_A.reshape(1, 3, color_raw_A.shape[-2], color_raw_A.shape[-1])
    color_A = torch.tensor(color_A)
    depth_A = depth_raw_A.reshape(1, 1, color_raw_A.shape[-2], color_raw_A.shape[-1])
    depth_A = torch.tensor(depth_A)
    # print for inspection
    # if size is not equal to resolution, resize
    if depth_A.shape[-1] != resolution:
        depth_A = torch.nn.functional.interpolate(depth_A, size=(resolution, resolution), mode='nearest')
    if color_A.shape[-1] != resolution:
        color_A = torch.nn.functional.interpolate(color_A, size=(resolution, resolution), mode='nearest')
    pose_raw_A = torch.tensor(pose_raw_A.astype(np.float32)).reshape(1, 4, 4)

    return depth_A, pose_raw_A, color_A, skip

def copy_frame(frame_id, img_path, output_path, resolution = 512):
    color_raw_path = os.path.join(img_path, 'colors', frame_id + '.png')
    # copy the file to output_path
    shutil.copy(color_raw_path, os.path.join(output_path, frame_id + '.png'))

def save_depth_with_colormap(depth_array, filename="depth_colormap.png", cmap="viridis"):
    """
    Saves a depth image as a PNG file with a colormap using Matplotlib.

    Args:
        depth_array (numpy.ndarray): The input depth map.
        filename (str): Output file name.
        cmap (str): Matplotlib colormap to use (e.g., 'plasma', 'jet', 'viridis').
    """
    # Normalize depth values to range [0, 1] for colormap
    depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array) + 1e-8)

    # Save using a colormap
    plt.imsave(filename, depth_normalized, cmap=cmap)
    # print(f"Saved depth image with colormap as {filename}")


class gen_data_fuser():
    def __init__(self, data_path, 
                 fusion_resolution=0.05, 
                 max_fusion_depth=3, 
                 fuse_color=True, 
                 depth_mask = 10,
                 front3d_base_matrix=front3d_base_matrix,
                 gt_condition = False,
                 gt_condition_path = None,
                 removal_threshold = 400
                 ):
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth
        self.fuse_color = fuse_color
        self.gen_data_path = data_path
        self.depth_mask = depth_mask
        self.front3d_base_matrix = front3d_base_matrix
        self.resolution = 256
        self.gt_condition = gt_condition
        self.condition_path = gt_condition_path
        self.removal_threshold = removal_threshold

        FOV = 90
        center = self.resolution / 2
        focal = self.resolution / 2 / np.tan(np.radians(FOV / 2))
        K = np.asarray(
                  [
                      [focal, 0.0, center],
                      [0.0, focal, center],
                      [0.0, 0.0, 1.0],
                    ],
                )
        self.K = torch.tensor(K).reshape(1, 3, 3)

    def fuse_tsdf_all(self, scene_name):
        self.fuser = Open3DFuser(self.gen_data_path, self.fusion_resolution, self.max_fusion_depth, self.fuse_color, front3d_base_matrix=front3d_base_matrix)
        print('fusing house:', scene_name)
        max_step = 4000
        scene_path = os.path.join(self.gen_data_path, scene_name)
        img_path = os.path.join(scene_path, 'colors')
        all_frames = os.listdir(img_path)
        output_path = os.path.join(scene_path, 'tsdf_fusion')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        step = 0
        for frame in all_frames:
            if '.png' not in frame or frame.startswith('.'):
                continue
            frame_id = frame.split('.')[0]
            print('fusing frame:', frame)
            depth_A, pose_raw_A, color_A, skip = load_frame(frame_id, scene_path, self.resolution)
            if skip:
                continue
            self.fuser.fuse_frames(depth_A, self.K, pose_raw_A, color_A, size= self.resolution)
            step += 1
            if step > max_step:
                break
        self.fuser.export_mesh(os.path.join(output_path, f'{scene_name}_grid{self.fusion_resolution}_depths{self.max_fusion_depth}_whole.ply'))
         
    def fuse_tsdf_step_by_step(self, scene_name):
        # find the json file
        json_path = os.path.join(self.gen_data_path, scene_name, f'sequence_{scene_name}.json')
        if not os.path.exists(json_path):
            print('no sequence json file found:', json_path)
            return
        with open(json_path, 'r') as f:
            data = json.load(f)
        sequence_list = data[scene_name]

        # make directory for the output
        output_path = os.path.join(self.gen_data_path, scene_name, 'sequence_tsdf')
        os.makedirs(output_path, exist_ok=True)
        output_path_cond = os.path.join(output_path, 'cond')
        output_path_target = os.path.join(output_path, 'target')
        output_path_mesh = os.path.join(output_path, f'mesh_grid{self.fusion_resolution}_depths{self.max_fusion_depth}')
        output_path_mesh_cond = os.path.join(output_path_mesh, 'cond')
        output_path_mesh_target = os.path.join(output_path_mesh, 'target')
        os.makedirs(output_path_mesh, exist_ok=True)
        os.makedirs(output_path_cond, exist_ok=True)
        os.makedirs(output_path_target, exist_ok=True)
        os.makedirs(output_path_mesh_cond, exist_ok=True)
        os.makedirs(output_path_mesh_target, exist_ok=True)

        #iterate through the sequence
        whole_fuser = Open3DFuser(self.gen_data_path, self.fusion_resolution, self.max_fusion_depth, self.fuse_color, front3d_base_matrix=front3d_base_matrix)
        for i in range(len(sequence_list)):
            frames_meta = sequence_list[i]
            print('fusing step:', i)
            frame_ids = frames_meta['frame_ids']
            conditional_frames = frame_ids['cond']
            target_frame = frame_ids['target']
            # todo: add methods to mask out depth method if it is not layout
            depth_methods = frames_meta['depth_method']

            # also make output directory for the step to save the images
            step_output_path_cond = os.path.join(output_path_cond, f'step_{i}_cond')
            os.makedirs(step_output_path_cond, exist_ok=True)
            step_output_path_target = os.path.join(output_path_target, f'step_{i}_target')
            os.makedirs(step_output_path_target, exist_ok=True)

            # fuse the conditional frames
            if self.gt_condition:
                cond_data_path = self.condition_path
            else:
                cond_data_path = self.gen_data_path
            cond_fuser = Open3DFuser(self.gen_data_path, self.fusion_resolution, self.max_fusion_depth, self.fuse_color, front3d_base_matrix=front3d_base_matrix)
            for frame in conditional_frames:
                frame_id = frame.split('.')[0]
                if not os.path.exists(os.path.join(cond_data_path, scene_name, 'colors', frame_id + '.png')):
                    print('frame not found:', frame_id)
                    continue
                depth_A, pose_raw_A, color_A, skip = load_frame(frame_id, os.path.join(cond_data_path, scene_name), self.resolution)
                if skip:
                    continue
                cond_fuser.fuse_frames(depth_A, self.K, pose_raw_A, color_A, size= self.resolution)
                # save the frames along the way
                copy_frame(frame_id, os.path.join(cond_data_path, scene_name), step_output_path_cond, self.resolution)

            cond_fuser.export_mesh(os.path.join(output_path_mesh_cond, f'step{i}_cond_{self.fusion_resolution}_{self.max_fusion_depth}.ply'))

            # fuse the target frame
            target_fuser = Open3DFuser(self.gen_data_path, self.fusion_resolution, self.max_fusion_depth, self.fuse_color, front3d_base_matrix=front3d_base_matrix)
            for frame in target_frame:
                frame_id = frame.split('.')[0]
                depth_A, pose_raw_A, color_A, skip = load_frame(frame_id, os.path.join(self.gen_data_path, scene_name), self.resolution)
                if skip:
                    continue
                target_fuser.fuse_frames(depth_A, self.K, pose_raw_A, color_A, size= self.resolution)
                whole_fuser.fuse_frames(depth_A, self.K, pose_raw_A, color_A, size= self.resolution)
                # save the frames along the way
                copy_frame(frame_id, os.path.join(self.gen_data_path, scene_name), step_output_path_target, self.resolution)
            target_fuser.export_mesh(os.path.join(output_path_mesh_target, f'step{i}_target_{self.fusion_resolution}_{self.max_fusion_depth}.ply'))
            whole_fuser.export_mesh(os.path.join(output_path_mesh, f'step{i}_whole_{self.fusion_resolution}_{self.max_fusion_depth}.ply'))

    def fuse_pcd_step_by_step(self, scene_name):
        # find the json file
        json_path = os.path.join(self.gen_data_path, scene_name, f'sequence_{scene_name}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        sequence_list = data[scene_name]

        # make directory for the output
        output_path = os.path.join(self.gen_data_path, scene_name, 'sequence_pcd')
        os.makedirs(output_path, exist_ok=True)
        output_path_cond = os.path.join(output_path, 'cond')
        output_path_target = os.path.join(output_path, 'target')
        output_path_pcd = os.path.join(output_path, f'pcd_dmask{self.depth_mask}')
        output_path_pcd_cond = os.path.join(output_path_pcd, 'cond')
        output_path_pcd_target = os.path.join(output_path_pcd, 'target')
        print('output_path_pcd:', output_path_pcd)
        os.makedirs(output_path_pcd, exist_ok=True)
        os.makedirs(output_path_cond, exist_ok=True)
        os.makedirs(output_path_target, exist_ok=True)
        os.makedirs(output_path_pcd_cond, exist_ok=True)
        os.makedirs(output_path_pcd_target, exist_ok=True)

        #iterate through the sequence
        for i in range(len(sequence_list)):
            frames_meta = sequence_list[i]
            print('fusing step:', i)
            frame_ids = frames_meta['frame_ids']
            conditional_frames = frame_ids['cond']
            target_frame = frame_ids['target']
            # todo: add methods to mask out depth method if it is not layout
            depth_methods = frames_meta['depth_method']

            # also make output directory for the step to save the images
            step_output_path_cond = os.path.join(output_path_cond, f'step_{i}_cond')
            os.makedirs(step_output_path_cond, exist_ok=True)
            step_output_path_target = os.path.join(output_path_target, f'step_{i}_target')
            os.makedirs(step_output_path_target, exist_ok=True)
            

            # fuse the conditional pcds
            if self.gt_condition:
                cond_data_path = self.condition_path
            else:
                cond_data_path = self.gen_data_path
            pcd_all = o3d.geometry.PointCloud()
            for frame in conditional_frames:
                frame_id = frame.split('.')[0]
                depth_A, pose_raw_A, color_A, skip = load_frame_for_pcd(frame_id, os.path.join(cond_data_path, scene_name), self.resolution)
                H,W = depth_A.shape
                # mask out any depths that are too far
                depth_A[depth_A > self.depth_mask] = 0
                coord = get_coord(H, W, self.K[0],pose_raw_A, depth_A)
                color = np.asarray(color_A)
                color = color[:,:,:3].reshape((H*W,3))
                coord = np.asarray(coord).reshape((H*W,3))
                # print('coord:', coord.shape, np.max(coord), np.min(coord))  

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coord)
                pcd.colors = o3d.utility.Vector3dVector(color)

                o3d.io.write_point_cloud(os.path.join(step_output_path_cond, f'{frame_id}.ply'), pcd)
                pcd_all = pcd+pcd_all

                copy_frame(frame_id, os.path.join(cond_data_path, scene_name), step_output_path_cond, self.resolution)

            o3d.io.write_point_cloud(os.path.join(output_path_pcd_cond, f'step{i}_cond_{self.depth_mask}.ply'), pcd_all)

            # fuse the target pcd
            pcd_all = o3d.geometry.PointCloud()
            for frame in target_frame:
                frame_id = frame.split('.')[0]
                depth_A, pose_raw_A, color_A, skip = load_frame_for_pcd(frame_id, os.path.join(self.gen_data_path, scene_name), self.resolution)
                H,W = depth_A.shape
                # mask out any depths that are too far
                depth_A[depth_A > self.depth_mask] = 0
                coord = get_coord(H,W,self.K[0],pose_raw_A, depth_A)
                color = np.asarray(color_A)
                color = color[:,:,:3].reshape((H*W,3))
                coord = np.asarray(coord).reshape((H*W,3))

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coord)
                pcd.colors = o3d.utility.Vector3dVector(color)

                o3d.io.write_point_cloud(os.path.join(step_output_path_target, f'{frame_id}.ply'), pcd)
                pcd_all = pcd+pcd_all

                copy_frame(frame_id, os.path.join(self.gen_data_path, scene_name), step_output_path_target, self.resolution)
            o3d.io.write_point_cloud(os.path.join(output_path_pcd_target, f'step{i}_target_{self.depth_mask}.ply'), pcd_all)

    def get_cleaned_mesh(self,scene_name):
        print('getting cleaned mesh for:', scene_name)
        # check if the entire mesh exits
        mesh_path = os.path.join(self.gen_data_path, scene_name, 'tsdf_fusion', f'{scene_name}_grid{self.fusion_resolution}_depths{self.max_fusion_depth}_whole.ply')
        if not os.path.exists(mesh_path):
            self.fuse_tsdf_all(scene_name)
        output_path = os.path.join(self.gen_data_path, scene_name, 'tsdf_fusion', f'{scene_name}_grid{self.fusion_resolution}_depths{self.max_fusion_depth}_whole_cleaned_at{self.removal_threshold}.ply')
        if os.path.exists(output_path):
            return
        mesh = o3d.io.read_triangle_mesh(mesh_path)
                
        cleaned_mesh = denoise_mesh_by_connectedComponents(mesh, removal_threshold=self.removal_threshold)
        
        # Check cleaned mesh bounds
        vertices_cleaned = np.asarray(cleaned_mesh.vertices)        
        # Remove vertices and faces above Y=2.0
        vertices = np.asarray(cleaned_mesh.vertices)
        triangles = np.asarray(cleaned_mesh.triangles)
        vertex_colors = np.asarray(cleaned_mesh.vertex_colors) if cleaned_mesh.has_vertex_colors() else None
        
        # Find vertices with Y <= 2.0
        valid_mask = vertices[:, 1] <= 2.0
        # Create a mapping from old vertex indices to new vertex indices
        old_to_new_idx = np.full(len(vertices), -1, dtype=int)
        old_to_new_idx[valid_mask] = np.arange(np.sum(valid_mask))
        
        # Filter vertices
        new_vertices = vertices[valid_mask]
        if vertex_colors is not None:
            new_vertex_colors = vertex_colors[valid_mask]
        
        # Filter triangles: keep only triangles where all three vertices are valid
        valid_triangles_mask = np.all(valid_mask[triangles], axis=1)        
        # Update triangle indices
        new_triangles = old_to_new_idx[triangles[valid_triangles_mask]]
        
        # Create new mesh
        filtered_mesh = o3d.geometry.TriangleMesh()
        filtered_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        filtered_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        if vertex_colors is not None and len(new_vertex_colors) > 0:
            filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(new_vertex_colors)
                
        # save the filtered mesh
        o3d.io.write_triangle_mesh(output_path, filtered_mesh)

    def make_nerf_json(self, scene_name, step_range = None):
        frame_meta = {
            "fl_x": 128, 
            "fl_y": 128,
            "cx": 128, 
            "cy": 128,
            "w": 256, 
            "h": 256,
            "frames": []
            }
        color_dir = os.path.join(self.gen_data_path, scene_name, 'colors')
        depth_dir = os.path.join(self.gen_data_path, scene_name, 'depth')
        cam_Ts_dir = os.path.join(self.gen_data_path, scene_name, 'cam_Ts')
        mask_dir = os.path.join(self.gen_data_path, scene_name, 'mask')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        all_files = os.listdir(color_dir)
        json_path = os.path.join(self.gen_data_path, scene_name, f'sequence_{scene_name}.json')
        if not os.path.exists(json_path):
            print('no sequence json file found:', json_path)
            return
        with open(json_path, 'r') as f:
            data = json.load(f)
        sequence_list = data[scene_name]
        if step_range is None:
            step_range = [0, (len(sequence_list))]
        for i in range(step_range[0], step_range[1]):
            frames_meta = sequence_list[i]
            frame_ids = frames_meta['frame_ids']
            conditional_frames = frame_ids['cond']
            target_frames = frame_ids['target']
            for frame_id in target_frames:
                color_path = os.path.join(color_dir, frame_id+'.png')
                depth_path = os.path.join(depth_dir, frame_id+'_rgb.npy')
                cam_Ts_path = os.path.join(cam_Ts_dir, frame_id+'.npy')
                if os.path.exists(color_path) and os.path.exists(depth_path) and os.path.exists(cam_Ts_path):
                    file_path = os.path.join('./colors', frame_id+'.png')
                    P = np.load(os.path.join(cam_Ts_dir, frame_id+'.npy'))
                    depth = np.load(os.path.join(depth_dir, frame_id+'_rgb.npy'))
                    mask = depth > self.depth_mask*1000
                    mask_bin = np.ones_like(mask)
                    mask_bin[mask] = 0
                    mask = depth < 0.001
                    mask_bin[mask] = 0
                    mask_bin = mask_bin.astype(np.uint8)
                    mask_bin = mask_bin*255
                    cv2.imwrite(os.path.join(mask_dir, frame_id+'.png'), mask_bin)
                    mask_path = os.path.join('./mask', frame_id+'.png')

                    depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
                    cv2.imwrite(os.path.join(self.gen_data_path, scene_name, 'depth', frame_id+'.png'), depth)
                    depth_file_path = os.path.join('./depth', frame_id+'.png')
                    
                    
                    frame_meta['frames'].append({
                        "file_path": file_path,
                        "depth_file_path": depth_file_path,
                        "mask_path": mask_path,
                        "transform_matrix": P.tolist()
                    })
        with open(os.path.join(self.gen_data_path, scene_name, 'transforms.json'), 'w') as f:
            json.dump(frame_meta, f, indent=4, separators=(',', ': '))

    def save_image_step_by_step(self, scene_name):
        # find the json file
        json_path = os.path.join(self.gen_data_path, scene_name, f'sequence_{scene_name}.json')
        if not os.path.exists(json_path):
            print('no sequence json file found:', json_path)
            return
        with open(json_path, 'r') as f:
            data = json.load(f)
        sequence_list = data[scene_name]

        # make directory for the output
        output_path = os.path.join(self.gen_data_path, scene_name, 'cond_target_images')
        os.makedirs(output_path, exist_ok=True)
        output_path_cond = os.path.join(output_path, 'cond')
        output_path_target = os.path.join(output_path, 'target')
        output_path_mesh = os.path.join(output_path, f'mesh_grid{self.fusion_resolution}_depths{self.max_fusion_depth}')
        output_path_mesh_cond = os.path.join(output_path_mesh, 'cond')
        output_path_mesh_target = os.path.join(output_path_mesh, 'target')
        os.makedirs(output_path_mesh, exist_ok=True)
        os.makedirs(output_path_cond, exist_ok=True)
        os.makedirs(output_path_target, exist_ok=True)
        os.makedirs(output_path_mesh_cond, exist_ok=True)
        os.makedirs(output_path_mesh_target, exist_ok=True)

        #iterate through the sequence
        for i in range(len(sequence_list)):
            frames_meta = sequence_list[i]
            print('fusing step:', i)
            frame_ids = frames_meta['frame_ids']
            conditional_frames = frame_ids['cond']
            target_frame = frame_ids['target']
            # todo: add methods to mask out depth method if it is not layout
            depth_methods = frames_meta['depth_method']

            # also make output directory for the step to save the images
            step_output_path_cond = os.path.join(output_path_cond, f'step_{i}_cond')
            os.makedirs(step_output_path_cond, exist_ok=True)
            step_output_path_target = os.path.join(output_path_target, f'step_{i}_target')
            os.makedirs(step_output_path_target, exist_ok=True)

            # fuse the conditional frames
            if self.gt_condition:
                cond_data_path = self.condition_path
            else:
                cond_data_path = self.gen_data_path
            for frame in conditional_frames:
                print('exporting conditional frame:', frame)
                frame_id = frame.split('.')[0]
                if not os.path.exists(os.path.join(cond_data_path, scene_name, 'colors', frame_id + '.png')):
                    print('cond_data_path:', cond_data_path)
                    print('frame not found:', frame_id)
                    continue
                depth_A, pose_raw_A, color_A, skip = load_frame(frame_id, os.path.join(cond_data_path, scene_name), self.resolution)
                if skip:
                    continue
                # save the frames along the way
                copy_frame(frame_id, os.path.join(cond_data_path, scene_name), step_output_path_cond, self.resolution)
                save_depth_with_colormap(depth_A[0,0].numpy(), os.path.join(step_output_path_cond, f'{frame_id}_depth.png'))


            # fuse the target frame
            for frame in target_frame:
                print('target_data_path:', self.gen_data_path)
                frame_id = frame.split('.')[0]
                depth_A, pose_raw_A, color_A, skip = load_frame(frame_id, os.path.join(self.gen_data_path, scene_name), self.resolution)
                if skip:
                    continue
                # save the frames along the way
                copy_frame(frame_id, os.path.join(self.gen_data_path, scene_name), step_output_path_target, self.resolution)
                save_depth_with_colormap(depth_A[0,0].numpy(), os.path.join(step_output_path_target, f'{frame_id}_depth.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse generated data into meshes')
    parser.add_argument('--gen_dir', type=str, default='../generated_data_v0',
                        help='Path to generated data directory')
    parser.add_argument('--gt_data_path', type=str, default='../generated_data_v0',
                        help='Path to ground truth data directory')
    parser.add_argument('--fusion_resolution', type=float, default=0.05,
                        help='TSDF fusion resolution')
    parser.add_argument('--max_fusion_depth', type=float, default=2.5,
                        help='Maximum fusion depth')
    parser.add_argument('--depth_mask', type=float, default=2.5,
                        help='Depth mask threshold')
    parser.add_argument('--removal_threshold', type=int, default=10,
                        help='Connected component removal threshold')
    parser.add_argument('--gt_condition', action='store_true',
                        help='Use ground truth condition data')
    args = parser.parse_args()

    gen_base = ''
    gen_path = os.path.join(gen_base, args.gen_dir)
    
    fuser = gen_data_fuser(gen_path, 
                           fusion_resolution=args.fusion_resolution, 
                           max_fusion_depth=args.max_fusion_depth, 
                           fuse_color=True, 
                           depth_mask=args.depth_mask,
                           front3d_base_matrix=front3d_base_matrix,
                           gt_condition=args.gt_condition,
                           gt_condition_path=args.gt_data_path,
                           removal_threshold=args.removal_threshold,
                           )
    scene_names = os.listdir(gen_path)
    for scene_name in scene_names:
        if scene_name.startswith('.'):
            continue
        fuser.fuse_tsdf_all(scene_name)
        fuser.get_cleaned_mesh(scene_name)
        print('finished cleaning mesh:', scene_name)
