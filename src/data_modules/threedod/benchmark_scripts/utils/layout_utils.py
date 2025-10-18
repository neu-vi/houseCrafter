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

def load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    return mesh

# Done: function to load bbox
def load_bboxes(gt_path):
    skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(gt_path)
    if skipped or boxes_corners.shape[0] == 0:
        return []
    boxes = box_utils.corners_to_boxes(boxes_corners)
    return boxes, boxes_corners

# Done: function to remove the artifacts by connectivity
def remove_artifacts(mesh, thres=0.1):
    print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    # find the largest cluster
    largest_cluster = np.argmax(cluster_n_triangles)
    threshold = cluster_n_triangles[largest_cluster] * thres
    print("Largest cluster size: ", cluster_n_triangles[largest_cluster])
    print("Threshold: ", threshold)

    cluster_area = np.asarray(cluster_area)
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
    mesh_0.remove_triangles_by_mask(triangles_to_remove)

    return mesh_0

# Done: Function to remove what was inside of the bounding boxes
def remove_mesh_inside_bboxes(mesh, bbox_corners):
    """
    Remove parts of the mesh whose vertices lie inside any of the bounding boxes.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        bbox_corners (np.ndarray): (M, 8, 3) array of 3D box corners.

    Returns:
        o3d.geometry.TriangleMesh: Filtered mesh with inner parts removed.
    """
    if len(bbox_corners) == 0:
        return mesh  # No boxes to remove

    # Convert vertices to NumPy array
    vertices = np.asarray(mesh.vertices)  # (N, 3)

    # Get per-point inclusion mask for each box
    mask_all = box_utils.points_in_boxes(vertices, bbox_corners)  # (N, M) bool array
    inside_mask = np.any(mask_all, axis=1)  # (N,) True if vertex is inside any box

    # Keep only vertices *not* inside any bbox
    keep_mask = ~inside_mask
    keep_indices = np.where(keep_mask)[0]

    # Find all triangles where all three vertices are kept
    triangles = np.asarray(mesh.triangles)  # (T, 3)
    triangle_mask = np.all(np.isin(triangles, keep_indices), axis=1)

    # Reindex vertices
    new_vertices = vertices[keep_indices]
    index_mapping = -np.ones(len(vertices), dtype=int)
    index_mapping[keep_indices] = np.arange(len(keep_indices))
    new_triangles = index_mapping[triangles[triangle_mask]]

    # Create new mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        new_colors = colors[keep_indices]
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    new_mesh.compute_vertex_normals()

    return new_mesh

def is_point_in_obb(point, obb):
    """
    Check if a point lies inside an Open3D OrientedBoundingBox.
    """
    R = obb.R
    center = obb.center
    extent = obb.extent / 2.0  # Half size in each dimension

    local_point = np.dot(R.T, (point - center))
    return np.all(np.abs(local_point) <= extent + 1e-6)  # Small epsilon for numerical stability

def keep_mesh_inside_obboxes(mesh, oboxes):
    """
    Keep only mesh triangles where all 3 vertices are inside any oriented bounding box.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    num_vertices = vertices.shape[0]

    # Create boolean mask: True if vertex is inside any box
    inside_mask = np.zeros(num_vertices, dtype=bool)

    for obb in oboxes:
        mask = np.array([is_point_in_obb(pt, obb) for pt in vertices])
        inside_mask |= mask  # Combine masks

    # Keep triangles where all 3 vertices are inside at least one box
    triangle_mask = np.all(inside_mask[triangles], axis=1)
    kept_triangles = triangles[triangle_mask]

    # Reindex vertices
    used_vertex_indices = np.unique(kept_triangles)
    index_remap = -np.ones(num_vertices, dtype=int)
    index_remap[used_vertex_indices] = np.arange(len(used_vertex_indices))

    new_vertices = vertices[used_vertex_indices]
    new_triangles = index_remap[kept_triangles]

    # Build mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        new_colors = colors[used_vertex_indices]
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    new_mesh.compute_vertex_normals()

    return new_mesh

# Done: function to detect walls/ceilings/floors
def detect_planes(mesh):
    # Use RANSAC or similar algorithm to detect planes
    mesh.compute_vertex_normals()
    skip_scene = False

    # Uniformly sample points from the mesh surface
    rest = mesh.sample_points_uniformly(number_of_points=500000)
    # using all defaults
    oboxes = rest.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    print("Detected {} patches".format(len(oboxes)))

    MIN_PLANE_AREA = 0.2  # square meters (adjust this based on your scene scale)

    # floor_planes = []
    # ceiling_planes = []
    other_planes = []
    max_floor_area = 0
    max_ceiling_area = 0

    up_vector = np.array([0, 0, 1])
    ceiling_planes = None
    floor_planes = None

    for obox in oboxes:
        # Get z-axis of the OBB (normal vector)
        R = obox.R  # Rotation matrix (3x3)
        z_axis = R[:, 2]  # Local z-axis points normal to the plane

        alignment = np.dot(z_axis, up_vector)
        area = np.prod(obox.extent[:2])
        print("Area: ", area)
        if area < MIN_PLANE_AREA:
            print("Area too small, skipping")
            continue  # Skip small planes


        if alignment > 0.9:
            if area > max_floor_area:
                floor_planes = obox
                max_floor_area = area
        elif alignment < -0.9:
            if area > max_ceiling_area:
                ceiling_planes = obox
                max_ceiling_area = area
        else:
            other_planes.append(obox)
    if ceiling_planes is None or floor_planes is None:
        print("No floor or ceiling detected, skipping")
        skip_scene = True

    # Return corners (M, 8, 3) and categorized planes
    # other_planes = stretch_planes_to_ceiling_height(floor_planes, ceiling_planes, other_planes)
    
    obox_corners = np.stack([np.asarray(obox.get_box_points()) for obox in other_planes], axis=0)
    obox_meshes = []
    for plane in other_planes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(plane, scale=[1, 1, 0.001])    
        obox_meshes.append(mesh)
    if not skip_scene:
        obox_meshes = stretch_box_meshes_to_ceiling_height(obox_meshes, floor_planes.center[2], ceiling_planes.center[2])
        for plane in [floor_planes, ceiling_planes]:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(plane, scale=[1, 1, 0.001])
            mesh.compute_vertex_normals()
            obox_meshes.append(mesh)
    
    return obox_corners, floor_planes, ceiling_planes, other_planes, obox_meshes, skip_scene

# todo4: function to check box visibility for each frame
def is_box_visible_in_view(box_corners, cam_pose, intrinsic, width, height):
    """
    Projects 3D corners into 2D image plane, checks if any are in frame.
    """
    P = intrinsic @ np.linalg.inv(cam_pose)[:3]  # Projection matrix
    corners_h = np.hstack([box_corners, np.ones((8, 1))])  # (8, 4)
    proj = (P @ corners_h.T).T  # (8, 3)

    proj[:, :2] /= proj[:, 2:3]  # Normalize

    x, y, z = proj[:, 0], proj[:, 1], proj[:, 2]
    in_front = z > 0
    in_view = (
        (x >= 0) & (x < width) &
        (y >= 0) & (y < height) &
        in_front
    )
    return np.any(in_view)

def save_depth_visualizations(depth_mesh_np, out_dir, frame_id, depth_np = None, prefix = None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    def save_visual(depth_array, name, prefix=None):
        # Normalize to [0, 1] for colormapping
        vmin, vmax = np.percentile(depth_array[depth_array > 0], [1, 99])
        norm_depth = np.clip((depth_array - vmin) / (vmax - vmin), 0, 1)

        # Save normalized grayscale
        if prefix is not None:
            name = f"{prefix}_{name}"
        # plt.imsave(os.path.join(out_dir, f"{name}_gray_{frame_id:04d}.png"),
        #            norm_depth, cmap='gray')

        # Save colormapped visualization
        plt.imsave(os.path.join(out_dir, f"{name}_jet_{frame_id:04d}.png"),
                   cm.jet(norm_depth))

    save_visual(depth_mesh_np, "depth_mesh", prefix)
    if depth_np is not None:
        save_visual(depth_np, "depth_orig", prefix)

def stretch_box_meshes_to_ceiling_height(box_meshes, floor_z, ceiling_z):
    stretched_meshes = []

    for mesh in box_meshes:
        # Step 1: Get all vertices in world space
        vertices = np.asarray(mesh.vertices)

        # Step 2: Compute current box center in XY (world coords)
        center_xy = vertices.mean(axis=0)[:2]  # (x, y)

        # Step 3: Get min and max z of the box
        min_z = vertices[:, 2].min()
        max_z = vertices[:, 2].max()

        current_height = max_z - min_z
        target_height = ceiling_z - floor_z

        if current_height < 1e-5:
            scale = 1.0  # avoid divide-by-zero
        else:
            scale = target_height / current_height

        # Step 4: Scale vertices along world Z
        # First shift to center at 0 in Z, scale, then re-center
        z_center = (min_z + max_z) / 2
        centered = vertices.copy()
        centered[:, 2] -= z_center
        centered[:, 2] *= scale
        centered[:, 2] += (floor_z + ceiling_z) / 2  # new center in Z

        # Step 5: Assign new vertices
        stretched = o3d.geometry.TriangleMesh()
        stretched.vertices = o3d.utility.Vector3dVector(centered)
        stretched.triangles = mesh.triangles
        stretched.compute_vertex_normals()
        stretched.paint_uniform_color([1, 0, 0])  # same as original

        stretched_meshes.append(stretched)

    return stretched_meshes

def is_image_rotated(cam_pose):
    # Camera up vector in camera space (assuming Open3D convention)
    cam_up = np.array([0, -1, 0])  # change to [0, 1, 0] if your convention differs

    # World up vectors to test against
    world_up_x = np.array([1, 0, 0])
    world_up_y = np.array([0, 1, 0])
    world_up_z = np.array([0, 0, 1])

    # Transform cam up to world space
    cam_up_world = cam_pose[:3, :3] @ cam_up
    cam_up_world /= np.linalg.norm(cam_up_world)

    # Check alignment with X or Y axis (which suggests vertical camera orientation)
    dot_x = abs(np.dot(cam_up_world, world_up_x))
    dot_y = abs(np.dot(cam_up_world, world_up_y))

    if max(dot_x, dot_y) > 0.9:
        print("image is rotated")
        return True  # rotated
    print("image is not rotated")
    return False  # not rotated
