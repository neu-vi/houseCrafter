import argparse
import numpy as np
import os
import time
import copy  

import utils.box_utils as box_utils
import utils.rotation as rotation
from utils.taxonomy import class_names, ARKitDatasetConfig
from utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt
import utils.visual_utils as visual_utils
import open3d as o3d
import json
import tqdm
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.layout_utils import (
    load_mesh,
    remove_artifacts,
    remove_mesh_inside_bboxes,
    keep_mesh_inside_obboxes,
    detect_planes,
    is_box_visible_in_view,
    save_depth_visualizations,
    stretch_box_meshes_to_ceiling_height,
    is_image_rotated
)
from scipy.ndimage import distance_transform_edt

class scene_renderer:
    def __init__(self, scene_id, 
                 data_root, 
                 output_dir,
                 width=640,
                 height=480,
                 camera_intrinsic=None,
                 ):
        self.scene_id = scene_id
        self.data_root = data_root
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.camera_intrinsic = camera_intrinsic
        self.label_json_pash = os.path.join(self.data_root, "label_record.json")
        # check if the scene is already in processed or invalid
        processed_scene_path = os.path.join(self.data_root, "processed_scene.json")
        if os.path.exists(processed_scene_path):
            with open(processed_scene_path, "r") as f:
                processed_scenes = json.load(f)
        else:
            processed_scenes = {}
        if self.scene_id in processed_scenes:
            print(f"Scene {self.scene_id} is already processed, skipping.")
            return
        invalid_scene_record = os.path.join(self.data_root, "invalid_scene_record.json")
        if os.path.exists(invalid_scene_record):
            with open(invalid_scene_record, "r") as f:
                invalid_scenes = json.load(f)
        else:
            invalid_scenes = []
        if self.scene_id in invalid_scenes:
            print(f"Scene {self.scene_id} is marked as invalid, skipping.")
            return
        print(f"Processing scene {self.scene_id}...")

        # get the loader for each frame
        data_path = os.path.join(self.data_root, self.scene_id, f"{self.scene_id}_frames")
        self.loader = TenFpsDataLoader(
            dataset_cfg=None,
            class_names=class_names,
            root_path=data_path,
        )
        total_frames = len(self.loader)
        # sample 5 frames from the loader
        random_indices = np.random.choice(total_frames, size=5, replace=False)
        acc = 0
        for i in random_indices:
            cam_pose_exp = self.loader[i]["pose"]
            is_rotated = is_image_rotated(cam_pose_exp)
            if is_rotated:
                acc += 1
        if acc > 3:
            print(f"Scene {self.scene_id} is invalid due to camera rotation, skipping.")
            self._mark_invalid_scene()
            return
        # Load the mesh
        mesh_path = os.path.join(data_root, scene_id, f"{scene_id}_3dod_mesh.ply")
        annotation_path = os.path.join(data_root, scene_id, f"{scene_id}_3dod_annotation.json")
        skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(annotation_path)
        self.bboxes_cornersq = boxes_corners
        self.bboxes_labels = labels
        self.bboxes_uids = uids
        self.bboxes_centers = centers
        self.bboxes_sizes = sizes

        # clean up the mesh to keep only the wall, floor and ceiling
        self.mesh = load_mesh(mesh_path)
        # Remove artifacts from the mesh
        self.mesh = remove_artifacts(self.mesh)
        self.mesh = remove_mesh_inside_bboxes(self.mesh, boxes_corners)
        o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True)

        # o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True)
        obox_corners, floor_planes, ceiling_planes, other_planes, bounding_meshes, skipped = detect_planes(self.mesh)
        if skipped:
            self._mark_invalid_scene()
            return  # Exit early

        self.bboxes_meshes = []
        for corner in self.bboxes_cornersq:
            box = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(corner)
                )
            # Step 2: Create an axis-aligned box mesh with the same dimensions
            box_mesh = o3d.geometry.TriangleMesh.create_box(
                width=box.extent[0],
                height=box.extent[1],
                depth=box.extent[2]
            )

            # Step 3: Move its center to the origin before rotating
            box_mesh.translate(-box_mesh.get_center())

            # Step 4: Rotate and translate to match the OBB
            box_mesh.rotate(box.R, center=(0, 0, 0))
            box_mesh.translate(box.center)
            # box_mesh = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
            box_mesh.paint_uniform_color([1, 0, 0])
            self.bboxes_meshes.append(box_mesh)
        # self.bboxes_meshes = stretch_box_meshes_to_ceiling_height(self.bboxes_meshes, floor_planes.center[2], ceiling_planes.center[2])

        self.floor_planes = floor_planes
        self.ceiling_planes = ceiling_planes
        self.wall_planes = other_planes
        self.bounding_meshes = bounding_meshes
        # self.mesh = keep_mesh_inside_obboxes(self.mesh, other_planes + [floor_planes] + [ceiling_planes])
        self.wall_mesh = copy.deepcopy(self.mesh)

        # process every 20 frames
        print(f"Rendering {len(self.loader)} frames...")
        for i in range(0, len(self.loader), 5):
            self.render_frame_one(i)
            # print(f"Rendered frame {i} of {len(self.loader)}")

        # Log as processed
        self._mark_processed_scene()

        # geometries = []
        # for obox in self.bounding_meshes:
        #     geometries.append(obox)
        # # for box_mesh in self.bboxes_meshes:
        # #     geometries.append(box_mesh)
        # geometries.append(self.mesh)
        # o3d.visualization.draw_geometries(geometries, mesh_show_back_face=True)

        # render_frame_one for every 10 frames in loader
        # for i in range(0, len(self.loader), 10):
        #     self.render_frame_one(i)
        #     print(f"Rendered frame {i} of {len(self.loader)}")

    def render_frame_one(self, frame_id):
        # Load the point cloud for the current frame
        frame = self.loader[frame_id]
        image = frame["image"]
        depth = frame["depth"]
        cam_pose = frame["pose"]
        cam_intrinsic_np = frame["intrinsics"]
        self.height, self.width = image.shape[:2]
        print('image shape: ', image.shape)
        print('image:', image[100:110,100:110])
        return

        # Set up intrinsic
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=self.width,
            height=self.height,
            fx=cam_intrinsic_np[0, 0],
            fy=cam_intrinsic_np[1, 1],
            cx=cam_intrinsic_np[0, 2],
            cy=cam_intrinsic_np[1, 2],
        )

        # Convert pose to extrinsic (Open3D expects camera-to-world)
        extrinsic = np.linalg.inv(cam_pose)
        # todo: test this part on the cluster later
        # self.render = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        # self.render.scene.set_background([0.0, 0.0, 0.0, 1.0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=False)
        vis.add_geometry(self.mesh)
        # self.material = o3d.visualization.rendering.MaterialRecord()
        # self.material.shader = "defaultLit"

        # === 1. Render mesh depth ===
        # Setup camera
        ctr = vis.get_view_control()
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = intrinsic
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

        # Update and render
        vis.poll_events()
        vis.update_renderer()

        # Capture depth
        depth_o3d = vis.capture_depth_float_buffer(do_render=True)
        # ?this is the depth we should use for the mesh
        depth_mesh_np = np.asarray(depth_o3d)

        color_o3d = vis.capture_screen_float_buffer(do_render=True)
        color_img = (np.asarray(color_o3d) * 255).astype(np.uint8)

        # Save _visualizations, TODO: COMMENT THIS OUT LATER
        output_dir = os.path.join(self.data_root, self.scene_id, self.output_dir, str(frame_id))
        os.makedirs(output_dir, exist_ok=True)
        save_depth_visualizations(depth_mesh_np, output_dir, frame_id, depth)
        # Save image
        imageio.imwrite(os.path.join(output_dir, f"image_{frame_id:04d}.png"), image)
        imageio.imwrite(os.path.join(output_dir, f"rendered_color_{frame_id:04d}.png"), color_img)
        vis.remove_geometry(self.mesh)
        vis.destroy_window()

        # === 2. Render each bbox if visible ===
        # Initialize list to collect depths and labels
        # ? depth and label info store here
        depth_list = []
        label_list = []

        # Load or initialize label-to-color mapping
        color_json_path = self.label_json_pash
        if os.path.exists(color_json_path):
            with open(color_json_path, "r") as f:
                label_to_color = json.load(f)
        else:
            label_to_color = {}
        # Keep a separate dict for label ordering
        label_to_index_path = os.path.join(self.data_root, "label_to_order.json")
        if os.path.exists(label_to_index_path):
            with open(label_to_index_path, "r") as f:
                label_to_index = json.load(f)
                current_index = max(label_to_index.values()) + 1
        else:
            label_to_index = {}
            current_index = 1

        for i, (bbox_mesh, bbox_corners) in enumerate(zip(self.bboxes_meshes, self.bboxes_cornersq)):
            if not is_box_visible_in_view(bbox_corners, cam_pose, cam_intrinsic_np, self.width, self.height):
                continue  # Skip invisible box
            label = self.bboxes_labels[i]

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=self.width, height=self.height, visible=False)
            vis.add_geometry(bbox_mesh)

            # Assign color
            if label in label_to_color:
                color = np.array(label_to_color[label])
            else:
                color = np.random.rand(3)
                label_to_color[label] = color.tolist()
                label_to_index[label] = current_index
                current_index += 1

            bbox_mesh.paint_uniform_color(color)

            ctr = vis.get_view_control()
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = intrinsic
            param.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()

            # Capture depth and color
            depth_o3d = vis.capture_depth_float_buffer(do_render=True)
            depth_bbox_np = np.asarray(depth_o3d)
            color_o3d = vis.capture_screen_float_buffer(do_render=True)
            color_img = (np.asarray(color_o3d) * 255).astype(np.uint8)

            # Save results
            # save_depth_visualizations(depth_bbox_np, output_dir, frame_id, None, prefix=f"bbox_{label}")
            imageio.imwrite(os.path.join(output_dir, f"box_{label}_color_{frame_id:04d}.png"), color_img)
            
            # Append depth and label to the list
            # ? this is the depth and label should use for the general bounding boxes
            depth_list.append(depth_bbox_np)
            label_list.append(label)

            vis.remove_geometry(bbox_mesh)
            vis.destroy_window()
        # update the json file
        with open(color_json_path, "w") as f:
            json.dump(label_to_color, f, indent=4)
        with open(label_to_index_path, "w") as f:
            json.dump(label_to_index, f, indent=4)
        # Save the depth and labels

        # === 3. the wall/floor/ceiling depths ===
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=False)
        # Add geometry (can be TriangleMesh or list of OBBs)
        if isinstance(self.bounding_meshes, list):
            for i, g in enumerate(self.bounding_meshes):
                color = np.random.rand(3)  # Default color
                g.paint_uniform_color(color)
                vis.add_geometry(g)
        else:
            color = np.random.rand(3)  # Default color
            self.bounding_meshes.paint_uniform_color(color)
            vis.add_geometry(self.bounding_meshes)
        # Set camera
        ctr = vis.get_view_control()
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = intrinsic
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        # Capture depth
        depth_o3d = vis.capture_depth_float_buffer(do_render=True)

        #? the depth we have for detected wall planes, use this for back up if the depth from mesh turn out to be null
        depth_wbox_np = np.asarray(depth_o3d)
        save_depth_visualizations(depth_wbox_np, output_dir, frame_id, None, prefix="wall_planes")
        # Also capture a color buffer for debugging
        color_o3d = vis.capture_screen_float_buffer(do_render=True)
        color_img = (np.asarray(color_o3d) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, f"walls_color_{frame_id:04d}.png"), color_img)
        vis.destroy_window()

        # Debug
        # print("depth mesh: ", depth_mesh_np.shape)
        # print("image: ", image.shape)
        # print("depth: ", depth.shape)
        # print("depth value range: ", depth_mesh_np[84,165] * 1000, depth[84,165], depth_wbox_np[84,165]*1000)
        # print("og depth value range: ", np.min(depth), np.max(depth))

        # todo: save the depth and layout info: 
        # === 4. get the depths for the boundaries(mesh) ===
        # ?walls: [d = 1, h, w, label], label = 0, d = depth
        # ?     for walls: first check the depth from mesh, if valid, use the the mesh_depth
        # ?     else, check the depth from wall plane & gt, 
        # ?           ->if plane_depth ~= gt_depth, use the gt_depth
        # ?           ->else, check if the space is occupied by furniture, if so, use the plane_depth
        # ?                 ->else, the area is empty, use an gt_depth to denote the space is empty
        # ?furniture: [d = 1, h, w, label], label = furniture id, d = depth
        H, W = self.height, self.width
        structural_depths = np.zeros((1, H, W), dtype=np.float32)
        TOLERANCE = 50
        depth_mesh_np = 1000 * depth_mesh_np
        depth_wbox_np = 1000 * depth_wbox_np

        # Masks
        valid_mesh = depth_mesh_np > 0
        valid_gt = depth > 0
        valid_wall = depth_wbox_np > 0
        wall_close_to_gt = np.abs(depth_wbox_np - depth) < TOLERANCE
        wall_further_than_gt = depth_wbox_np > depth + TOLERANCE
        wall_closer_than_gt = depth_wbox_np < depth - TOLERANCE

        # 1. Mesh depth is valid → use it as wall
        structural_depths[0][valid_mesh] = depth_mesh_np[valid_mesh]
        # 2. Where mesh is invalid, and wall agrees with GT → use GT as wall
        use_gt_wall = ~valid_mesh & valid_gt & valid_wall & wall_close_to_gt
        structural_depths[0][use_gt_wall] = depth[use_gt_wall]
        # 3. Where mesh is invalid, wall doesn't match GT(and is further) → use wall as wall
        use_wall_when_furniture_blocks_gt = (
            ~valid_mesh & valid_wall & wall_further_than_gt
        )
        structural_depths[0][use_wall_when_furniture_blocks_gt] = depth_wbox_np[use_wall_when_furniture_blocks_gt]
        # 4. Where mesh is invalid, wall doesn't match GT(and the wall is closer than gt) → use GT as wall
        use_gt_for_empty_space = (
            ~valid_mesh & valid_wall & wall_closer_than_gt
        )
        structural_depths[0][use_gt_for_empty_space] = depth[use_gt_for_empty_space]
        # 5. if there is still an invalid area, interpolate the depth

        valid = structural_depths[0] > 0
        if not np.all(valid):  # only run if there are invalid pixels
            # Find nearest valid pixel for each invalid pixel
            distance, indices = distance_transform_edt(~valid, return_indices=True)
            structural_depths[0][~valid] = structural_depths[0][tuple(indices[:, ~valid])]
        
        # save visualizations
        save_depth_visualizations(structural_depths[0], output_dir, frame_id, None, prefix="structural_depths_final")

        # todo, merge and save the depth and layout_label info
        # ? layout_pos = [n, h, w], where n = number of visible bounding boxes, the first should be the wall, then followed by furniture boxes, the value in the channel is the depth
        # ? layout_label = [n, h, w], where n = number of visible bounding boxes, the first should be the wall, then followed by furniture boxes, the value in the channel is the label id(label id = 0 for wall, label id = i+1 for furniture i by its order in the json file)
        layout_pos = np.zeros((len(self.bboxes_meshes) + 1, H, W), dtype=np.float32)
        layout_label = np.zeros((len(self.bboxes_meshes) + 1, H, W), dtype=np.int32)
        layout_pos[0] = structural_depths[0]
        layout_label[0] = 0
        for i, depth in enumerate(depth_list):
            layout_pos[i + 1] = depth_list[i]
            valid_pixels = depth_list[i] > 0
            label = label_list[i]
            label_index = label_to_index[label]
            layout_label[i + 1][valid_pixels] = label_index
        # Save layout_pos and layout_label
        layout_pos_path = os.path.join(output_dir, f"layout_pos_{frame_id:04d}.npy")
        layout_label_path = os.path.join(output_dir, f"layout_label_{frame_id:04d}.npy")
        np.save(layout_pos_path, layout_pos)
        np.save(layout_label_path, layout_label)

    def _mark_invalid_scene(self):
        self.valid = False
        invalid_scene_record = os.path.join(self.data_root, "invalid_scene_record.json")
        if os.path.exists(invalid_scene_record):
            with open(invalid_scene_record, "r") as f:
                invalid_scenes = json.load(f)
        else:
            invalid_scenes = []

        if self.scene_id not in invalid_scenes:
            invalid_scenes.append(self.scene_id)
            with open(invalid_scene_record, "w") as f:
                json.dump(invalid_scenes, f, indent=4)
        print(f"Scene {self.scene_id} is invalid, skipping rendering.")

    def _mark_processed_scene(self):
        processed_scene_path = os.path.join(self.data_root, "processed_scene.json")
        if os.path.exists(processed_scene_path):
            with open(processed_scene_path, "r") as f:
                processed_scenes = json.load(f)
        else:
            processed_scenes = {}

        if self.scene_id not in processed_scenes:
            processed_scenes[self.scene_id] = {"status": "processed"}
            with open(processed_scene_path, "w") as f:
                json.dump(processed_scenes, f, indent=4)



def main():
    print("Loading scene renderer")
    parser = argparse.ArgumentParser(description="Render scene with mesh and depth")
    parser.add_argument("--scene_id", type=str, required=True, help="Scene ID")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for rendered frames")

    args = parser.parse_args()
    # Set up the scene renderer
    renderer = scene_renderer(
        scene_id=args.scene_id,
        data_root=args.data_root,
        output_dir=args.output_dir
    )

    # scenes_to_process = os.listdir(args.data_root)
    # i = 0
    # for scene in scenes_to_process:
    #     print(f"Processing scene: {scene}")
    #     renderer = scene_renderer(scene, args.data_root, args.output_dir)
    #     print(f"processing scene {i} of {len(scenes_to_process)}")
    #     i += 1
    # # Add rendering logic here

if __name__ == "__main__":
    main()