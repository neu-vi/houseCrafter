# Revised from
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler

from collections import defaultdict
import numpy as np
import json
import os
import pickle
from tqdm import tqdm
from typing import List, Dict
from visualization.utils.tools import R_from_pitch_yaw_roll, get_box_corners, project_points_to_2d

from .threed_front_scene import Asset, ModelInfo, Room, ThreedFutureModel, \
    ThreedFutureExtra

FRONT3D2BLENDER = np.array([
    [1.0, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

def parse_threed_front_scenes(
    dataset_directory, path_to_model_info, path_to_models, path_to_scenes,
    path_to_room_masks_dir=None, json_files='all'
):
    # Parse the model info
    mf = ModelInfo.from_file(path_to_model_info)
    model_info = mf.model_info


    if json_files == 'all':
        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")]
    else:
        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(json_files)]

    scenes = []
    # Start parsing the dataset
    print("Loading dataset")
    for i, m in enumerate(tqdm(path_to_scene_layouts)):
        save_scene_path = os.path.join(path_to_scenes, os.path.splitext(os.path.basename(m))[0] + '.pkl')
        if os.path.exists(save_scene_path):
            scenes.extend(pickle.load(open(save_scene_path, "rb")))
            continue

        with open(m) as f:
            data = json.load(f)

        # Parse the furniture of the scene
        furniture_in_scene = defaultdict()
        for ff in data["furniture"]:
            # TODO: shall we consider an object that only has a bounding box without a shape model?
            if "valid" in ff and ff["valid"]:
                furniture_in_scene[ff["uid"]] = dict(
                    model_uid=ff["uid"],
                    model_jid=ff["jid"],
                    model_info=model_info[ff["jid"]])

        # Parse the extra meshes of the scene e.g walls, doors,
        # windows etc.
        meshes_in_scene = defaultdict()
        for mm in data["mesh"]:
            meshes_in_scene[mm["uid"]] = dict(
                mesh_uid=mm["uid"],
                mesh_jid=mm["jid"],
                mesh_xyz=np.asarray(mm["xyz"]).reshape(-1, 3),
                mesh_faces=np.asarray(mm["faces"]).reshape(-1, 3),
                mesh_type=mm["type"]
            )

        # Parse the rooms of the scene
        scene = data["scene"]
        # Keep track of the parsed rooms
        rooms = []
        for rr in scene["room"]:
            # Keep track of the furniture in the room
            furniture_in_room = []
            # Keep track of the extra meshes in the room
            extra_meshes_in_room = []

            for cc in rr["children"]:
                if cc["ref"] in furniture_in_scene:
                    tf = furniture_in_scene[cc["ref"]]
                    furniture_in_room.append(ThreedFutureModel(
                       tf["model_uid"],
                       tf["model_jid"],
                       tf["model_info"],
                       cc["pos"],
                       cc["rot"],
                       cc["scale"],
                       path_to_models
                    ))
                elif cc["ref"] in meshes_in_scene:
                    mf = meshes_in_scene[cc["ref"]]
                    extra_meshes_in_room.append(ThreedFutureExtra(
                        mf["mesh_uid"],
                        mf["mesh_jid"],
                        mf["mesh_xyz"],
                        mf["mesh_faces"],
                        mf["mesh_type"],
                        cc["pos"],
                        cc["rot"],
                        cc["scale"]
                    ))
                else:
                    continue
            if len(furniture_in_room) > 1:
                # Add to the list
                rooms.append(Room(
                    rr["instanceid"],                # room_id
                    rr["type"].lower(),              # room_type
                    furniture_in_room,               # bounding boxes
                    extra_meshes_in_room,            # extras e.g. walls
                    m.split("/")[-1].split(".")[0],  # json_path
                    path_to_room_masks_dir
                ))
        pickle.dump(rooms, open(save_scene_path, "wb"))
        scenes.extend(rooms)

    return scenes


def parse_threed_future_models(
    dataset_directory, path_to_models, path_to_model_info, path_to_3d_future_objects
):
    if os.path.exists(path_to_3d_future_objects):
        furnitures = pickle.load(
            open(path_to_3d_future_objects, "rb")
        )
    else:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        model_info = mf.model_info

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")
        ]
        # List to keep track of all available furniture in the dataset
        furnitures = []
        unique_furniture_ids = set()

        # Start parsing the dataset
        print("Loading dataset")
        for i, m in enumerate(tqdm(path_to_scene_layouts)):
            with open(m) as f:
                data = json.load(f)
                # Parse the furniture of the scene
                furniture_in_scene = defaultdict()
                for ff in data["furniture"]:
                    if "valid" in ff and ff["valid"]:
                        furniture_in_scene[ff["uid"]] = dict(
                            model_uid=ff["uid"],
                            model_jid=ff["jid"],
                            model_info=model_info[ff["jid"]]
                        )
                # Parse the rooms of the scene
                scene = data["scene"]
                for rr in scene["room"]:
                    for cc in rr["children"]:
                        if cc["ref"] in furniture_in_scene:
                            tf = furniture_in_scene[cc["ref"]]
                            if tf["model_uid"] not in unique_furniture_ids:
                                unique_furniture_ids.add(tf["model_uid"])
                                furnitures.append(ThreedFutureModel(
                                    tf["model_uid"],
                                    tf["model_jid"],
                                    tf["model_info"],
                                    cc["pos"],
                                    cc["rot"],
                                    cc["scale"],
                                    path_to_models
                                ))
                        else:
                            continue
        pickle.dump(furnitures, open(path_to_3d_future_objects, "wb"))

    return furnitures

def parse_inst_from_3dfront(inst_dict:Dict, rooms:List, inst_rm_uid:str) -> Dict:
    '''
    Get 3D instance information from blender rendering info.
    :param inst_dict: an instance from blender rendered images.
    :param rooms: processed rooms from 3D-Front.
    :param inst_rm_uid: the room uid of inst_dict
    :return:
    '''
    target_room = next((rm for rm in rooms if rm.uid == inst_rm_uid), None)
    if target_room is None:
        return {'model_path': None, 'bbox3d': None}
    target_insts = [box for box in target_room.bboxes if box.model_uid == inst_dict['uid'] and box.model_jid == inst_dict['jid']]
    if not target_insts:
        print('there is no 3D object in 3dfront furniture corresponds to an 2D inst in blender.')
        return {'model_path': None, 'bbox3d': None}
    elif len(target_insts) == 1:
        inst = target_insts[0]
    else:
        positions = np.array([box.position for box in target_insts])
        # transfer from blender coordinates to 3dfront coordinates
        target_inst_location = np.array(inst_dict['location'])
        target_inst_location = np.linalg.inv(FRONT3D2BLENDER[:3,:3]) @ target_inst_location
        pairwise_dists = np.linalg.norm(positions - target_inst_location, axis=1)
        inst_idx = pairwise_dists.argmin()
        # if there are no objects in 3dfront processed objects for each inst in blender.
        if pairwise_dists[inst_idx] > 1e-2:
            print('there is no 3D object in 3dfront furniture corresponds to an 2D inst in blender.')
            return {'model_path': None, 'bbox3d': None}
        else:
            inst = target_insts[inst_idx]
    try:
        bbox_params = np.concatenate([inst.centroid(), inst.size, inst.get_world_orientation()])
    except:
        print('Wrong labeled object.')
        return {'model_path': None, 'bbox3d': None}

    return {'model_path': inst.raw_model_path, 'bbox3d': bbox_params}

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_room_uid_from_rendering(inst_info: List):
    room_uid_stat = {inst['room_uid']: 0 for inst in inst_info}
    if len(room_uid_stat.keys()) == 1:
        return inst_info[0]['room_uid']
    else:
        for inst in inst_info:
            area = np.count_nonzero(inst['mask'])
            room_uid_stat[inst['room_uid']] += area
        target_room_uid = None
        max_area = 0
        for room_uid, area in room_uid_stat.items():
            if area > max_area:
                max_area = area
                target_room_uid = room_uid
        return target_room_uid


def project_insts_to_2d(inst_info, cam_K, cam_T):
    '''project vertices of 3D bboxes to image plane.'''
    if len(cam_T.shape) == 3:
        cam_T = cam_T[0]
    projected_boxes = []
    for inst in inst_info:
        if not isinstance(inst['bbox3d'], np.ndarray):
            projected_boxes.append(None)
        elif not np.issubdtype(inst['bbox3d'].dtype, np.floating):
            projected_boxes.append(None)
        else:
            centroid = inst['bbox3d'][:3]
            R_mat = R_from_pitch_yaw_roll(0., inst['bbox3d'][6], 0.)[0]
            vectors = np.diag(np.array(inst['bbox3d'][3:6]) / 2.).dot(R_mat.T)
            box_corners = np.array(get_box_corners(centroid, vectors))

            pixels = project_points_to_2d(box_corners, cam_K, cam_T)
            projected_boxes.append(pixels[:, :2])
    return projected_boxes

def get_inst_spatial_scope(inst_info, padding=0):
    min_xyz = np.array([1e5, 1e5, 1e5])
    max_xyz = np.array([-1e5, -1e5, -1e5])
    for inst in inst_info:
        centroid = inst['bbox3d'][:3]
        R_mat = R_from_pitch_yaw_roll(0., inst['bbox3d'][6], 0.)[0]
        vectors = np.diag(np.array(inst['bbox3d'][3:6]) / 2.).dot(R_mat.T)
        box_corners = np.array(get_box_corners(centroid, vectors))
        min_xyz = np.minimum(np.min(box_corners, axis=0), min_xyz)
        max_xyz = np.maximum(np.max(box_corners, axis=0), max_xyz)
    min_xyz = min_xyz - padding
    max_xyz = max_xyz + padding
    return min_xyz, max_xyz