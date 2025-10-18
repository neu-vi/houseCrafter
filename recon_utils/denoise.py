import os
import open3d as o3d
import numpy as np
import copy


def denoise_mesh_by_connectedComponents(mesh, voxel_size=0.02, removal_threshold=50):
    print('culling ceiling area')
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    # print('vertice ranges:', vertices.min(axis=0), vertices.max(axis=0))
    mask = vertices[:, 1] > 2.3
    mesh.remove_vertices_by_mask(mask)
    vertices = np.asarray(mesh.vertices)
    mask = vertices[:, 1] < -0.4
    mesh.remove_vertices_by_mask(mask)
    # print('vertice ranges:', vertices.min(axis=0), vertices.max(axis=0))
    
    
    print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print("Show mesh with small clusters removed")
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < removal_threshold
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    # o3d.visualization.draw_geometries([mesh_0])
    return mesh_0


# path_to_raw = '../../Ours_meshes'
# path_to_denoised = '../../Ours_meshes_denoised'
# scene_name = '6a03badb-61ab-4e22-9564-de2fd2ed1938'

# for scene_name in os.listdir(path_to_raw):
#     print(scene_name)
#     mesh = o3d.io.read_triangle_mesh(os.path.join(path_to_raw, scene_name))
#     denoised = denoise_mesh_by_connectedComponents(mesh)
#     o3d.io.write_triangle_mesh(os.path.join(path_to_denoised, scene_name), denoised)
# mesh = o3d.io.read_triangle_mesh(os.path.join(path_to_raw, scene_name + '.ply'))
# denoised = denoise_mesh_by_connectedComponents(mesh)
# o3d.io.write_triangle_mesh(os.path.join(path_to_denoised, scene_name + '.ply'), denoised)

if __name__ == "__main__":
    path_to_raw = '../cherry_meshes'
    path_to_denoised = '..'
    scene_name = '644dba87-0d65-4897-912c-38185791b3c2'
    mesh = o3d.io.read_triangle_mesh(os.path.join(path_to_raw, scene_name + '.ply'))
    denoised = denoise_mesh_by_connectedComponents(mesh)
    o3d.io.write_triangle_mesh(os.path.join(path_to_denoised, scene_name + '_denoised.ply'), denoised)

    # for scene_name in os.listdir(path_to_raw):
    #     print(scene_name)
    #     mesh = o3d.io.read_triangle_mesh(os.path.join(path_to_raw, scene_name))
    #     denoised = denoise_mesh_by_connectedComponents(mesh)
    #     o3d.io.write_triangle_mesh(os.path.join(path_to_denoised, scene_name), denoised)
