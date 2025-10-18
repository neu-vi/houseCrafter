"""
write class to to load and represent the scene
follow the tool box: scene room, furniture, instance, layout (non furniture mesh)

pointcloud (from rendered images), camera poses

visualizer: open3d load class palate
"""

import os
import pickle

import numpy as np
from front3d_scene import load_labels, load_scene
from front3d_viz import Front3DViz

# scene_id = "0b1953f7-3bab-4a2e-b0c8-396d0170d6b0"
# scene_id = "0ec7ce97-e93d-4842-9c89-b947136bb393"
scene_id = "0a8d471a-2587-458a-9214-586e003e9cf9"
# scene_id = "6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9"
# scene_id = "03279ac8-717b-4b94-96e1-3b6a7e94e782"
# scene_id = "0a8d471a-2587-458a-9214-586e003e9cf9"
# scene_id = "00ad8345-45e0-45b3-867d-4a3c88c2517a"
scene_id = "00004f89-9aa5-43c2-ae3c-129586be8aaa"
# scene_id = "0032b185-4914-49e5-b973-f82271674308"

scene_dir = "/mnt/DATA/personal_projects/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT"
model_dir = "/mnt/DATA/personal_projects/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model"
# funiture_bbox_dir = "/media/hieu/T7/3d-front/model_bbox"
label_id_mapping_file = "/mnt/DATA/personal_projects/BlenderProc-3DFront/blenderproc/resources/front_3D/blender_label_mapping.csv"
model_info_file = "/mnt/DATA/personal_projects/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model/model_info_revised.json"
# output_layout_path = f"/media/hieu/T7/3d-front/layouts/{scene_id}.json"

# scene_dir = "/mnt/Data/hieu/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT"
# model_dir = "/mnt/Data/hieu/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model"
# locations_dir = "/media/hieu/T7/3dfront_render/locations_secondbedroom"
# pose_dir = "/media/hieu/T7/3dfront_render/poses_secondbedroom_015"

funiture_bbox_dir = "/media/hieu/T7/3d-front/model_bbox"
layout_folder = "/media/hieu/T7/3dfront_render/layout_test2"
layout_folder = "/media/hieu/T7/3dfront_render/layout_from_pose"
obj_pose_dir = "/media/hieu/T7/3d-front/obj_poses_100"
BASE_MATRIX = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)

# scene_ids = os.listdir("/media/hieu/T7/3d-front/3D-FRONT")
# scene_ids = sorted([f.split(".")[0] for f in scene_ids if f.endswith(".json")])
scene_ids = [scene_id]
for i, scene_id in enumerate(scene_ids[:100]):
    # if i < 30:
    #     continue
    print(i, scene_id)
    scene_file = os.path.join(scene_dir, f"{scene_id}.json")
    # os.makedirs(locations_dir, exist_ok=True)
    try:
        house = load_scene(scene_file, model_dir, funiture_bbox_dir)
    except Exception as e:
        print(e)
        continue
    # locations = house.sample_locations(
    #     dist=0.15, variance=0.1, exclude_furniture=True, seed=0, height_range=[1.2, 1.7],
    #     wall_margin=0.1, furniture_margin=0.1
    # )
    # print(len(locations))
    # location_path = os.path.join(locations_dir, f"{scene_id}.npy")
    # np.save(location_path, locations)

    # poses = house.sample_orientations(locations, seed=0)
    # os.makedirs(pose_dir, exist_ok=True)
    # pose_path = os.path.join(pose_dir, f"{scene_id}.npy")
    # np.save(pose_path, poses)

    label_id_mapping, model_id_to_label, label_id2name = load_labels(
        label_id_mapping_file, model_info_file
    )
    layout_pcd_path = os.path.join(layout_folder, scene_id)
    # house.save_furniture_2d_boxes(label_id_mapping, model_id_to_label, output_layout_path)
    # print(len(locations))
    # grid = np.stack([xz[:, 0], np.ones(xz.shape[0]) * height, xz[:, 1]], axis=1)
    # viz = Front3DViz(house, locations=locations)
    viz = Front3DViz(
        house, label_id2name=label_id2name, layout_pcd_path=layout_pcd_path
    )
    # viz = Front3DViz(house)
    poses_dict = pickle.load(open(f"{obj_pose_dir}/{scene_id}.pkl", "rb"))
    print(len(poses_dict))
    for k, v in poses_dict.items():
        # if not "bed" in v["room"].lower():
        #     continue
        print(k, v["room"])
        poses = v["poses"]
        poses[:, :3, :3] = poses[:, :3, :3] @ BASE_MATRIX
        viz.show(
            ["furniture", "non_furniture", "camera_poses", "3dbox"],
            ceiling=False,
            c2w=poses,
        )
        print()
    # viz.show(["furniture", "layout_pcd"], ceiling=False, class_id=32)
    # viz.show(["furniture"], ceiling=False)
    # viz.show(["non_furniture", "locations", "3dbox", "furniture"], ceiling=False)
    # viz.show(["floor"], ceiling=False)
    # viz.show(["non_furniture", "3dbox"], ceiling=False)
    # viz.show(["non_furniture", "2dlayout"], ceiling=False, floor=False)
    print()
