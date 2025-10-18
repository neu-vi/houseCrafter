import blenderproc as bproc
import argparse
import os
import numpy as np
from pathlib import Path
import json
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file folder")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument("output_folder", help="Path to where the data should be saved")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=480, help="Image width.")
    parser.add_argument("--res_y", type=int, default=360, help="Image height.")
    return parser.parse_args()

def get_folders(args):
    front_folder = Path(args.front_folder)
    future_folder = Path(args.future_folder)
    front_3D_texture_folder = Path(args.front_3D_texture_folder)
    output_folder = Path(args.output_folder)
    return front_folder, future_folder, front_3D_texture_folder, output_folder

def check_name(name, category_name):
    return True if category_name in name.lower() else False

if __name__ == '__main__':
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, output_folder = get_folders(args)

    if not front_folder.exists() or not future_folder.exists() or not front_3D_texture_folder.exists():
        raise Exception("One of these folders does not exist!")

    bproc.init()
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # set the light bounces
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                     transmission_bounces=200, transparent_max_bounces=200)
    # set intrinsic parameters
    bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x, image_height=args.res_y,
                                                    lens_unit="FOV")

    cam_K = bproc.camera.get_intrinsics_as_K_matrix()

    # read 3d future model info
    with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
        model_info_data = json.load(f)
    model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' for m in
                          model_info_data}

    front_json = next(front_folder.iterdir())

    scene_name = front_json.name[:-len(front_json.suffix)]

    # load the front 3D objects
    loaded_objects = bproc.loader.load_front3d(
        json_path=str(front_json),
        future_model_path=str(future_folder),
        front_3D_texture_path=str(front_3D_texture_folder),
        label_mapping=mapping,
        model_id_to_label=model_id_to_label)

    #-------------------------------------------------------------------------
    #          Sample camera extrinsics
    # -------------------------------------------------------------------------
    # Init sampler for sampling locations inside the loaded front3D house
    point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

    # Init bvh tree containing all mesh objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(
        [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

    # filter some objects from the loaded objects, which are later used in calculating an interesting score
    interest_score_setting = {'ceiling': 0, 'column': 0, 'customizedpersonalizedmodel': 0, 'beam': 0, 'wallinner': 0,
                              'slabside': 0, 'customizedfixedfurniture': 0, 'cabinet/lightband': 0, 'window': 0,
                              'hole': 0, 'customizedplatform': 0, 'baseboard': 0, 'customizedbackgroundmodel': 0,
                              'front': 0, 'walltop': 0, 'wallouter': 0, 'cornice': 0, 'sewerpipe': 0,
                              'smartcustomizedceiling': 0, 'customizedfeaturewall': 0, 'customizedfurniture': 0,
                              'slabtop': 0, 'baywindow': 0, 'door': 0, 'customized_wainscot': 0, 'slabbottom': 0,
                              'back': 0, 'flue': 0, 'extrusioncustomizedceilingmodel': 0,
                              'extrusioncustomizedbackgroundwall': 0, 'floor': 0, 'lightband': 0,
                              'customizedceiling': 0, 'void': 0, 'pocket': 0, 'wallbottom': 0, 'chair': 10, 'sofa': 10,
                              'table': 10, 'bed': 10}
    special_objects = []
    special_object_scores = {}
    for category_name, category_score in interest_score_setting.items():
        special_objects_per_category = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name(), category_name)]
        special_objects.extend(special_objects_per_category)
        unique_cat_ids = set(special_objects_per_category)
        for cat_id in unique_cat_ids:
            special_object_scores[cat_id] = category_score

    # sample camera poses
    proximity_checks = {}
    cam_Ts = []
    n_cameras = 10
    floor_areas = np.array(point_sampler.get_floor_areas())
    cam_nums = np.ceil(floor_areas / floor_areas.sum() * n_cameras).astype(np.int16)
    n_tries = 20

    for floor_id, cam_num_per_scene in enumerate(cam_nums):
        cam2world_matrices = []
        coverage_scores = []
        tries = 0
        while tries < n_tries:
            # sample cam loc inside house
            height = np.random.uniform(1.4, 1.8)
            location = point_sampler.sample_by_floor_id(height, floor_id=floor_id)
            # Sample rotation (fix around X and Y axis)
            rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])  # pitch, roll, yaw
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

            # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
            # meters and make sure that no background is visible, finally make sure the view is interesting enough
            obstacle_check = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree)
            coverage_score = bproc.camera.scene_coverage_score(cam2world_matrix, special_objects,
                                                               special_objects_weight=special_object_scores)
            # for sanity check
            if obstacle_check and coverage_score >= 0.5:
                cam2world_matrices.append(cam2world_matrix)
                coverage_scores.append(coverage_score)
                tries += 1
        cam_ids = np.argsort(coverage_scores)[-cam_num_per_scene:]
        for cam_id, cam2world_matrix in enumerate(cam2world_matrices):
            if cam_id in cam_ids:
                bproc.camera.add_camera_pose(cam2world_matrix)
                cam_Ts.append(cam2world_matrix)

    # render the whole pipeline
    data = bproc.renderer.render()
    default_values = {"location": [0, 0, 0], "cp_inst_mark": '', "cp_uid": '', "cp_jid": '', "cp_room_id": ""}
    data.update(bproc.renderer.render_segmap(map_by=["instance", "class", "cp_uid", "cp_jid", "cp_inst_mark" , "cp_room_id", "location"],
                                             default_values=default_values))

    # write camera intrinsics
    np.save(output_folder.joinpath('cam_K.npy'), cam_K)

    # write camera extrinsics
    data['cam_Ts'] = cam_Ts
    # write the data to a .hdf5 container
    scene_output_folder = output_folder.joinpath(scene_name)
    bproc.writer.write_hdf5(str(scene_output_folder), data)

