import sys
sys.path.append('.')
import argparse
import h5py
import numpy as np
import json
from visualization.front3d import Threed_Front_Config
from visualization.front3d.tools.threed_front import ThreedFront
from visualization.front3d.vis_classes import VIS_3DFRONT, VIS_3DFRONT_2D
from visualization.front3d.tools.utils import parse_inst_from_3dfront, project_insts_to_2d
from visualization.utils.tools import label_mapping_2D

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a 3D-FRONT room.")
    parser.add_argument("--json_file", type=str, default='6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json',
                        help="The json file of the property in 3D-Front to be visualized.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # initialize category labels and mapping dict for specific room type.
    dataset_config = Threed_Front_Config()
    dataset_config.init_generic_categories_by_room_type('all')

    '''Read 3D-Front Data'''
    json_files = [args.json_file]
    d = ThreedFront.from_dataset_directory(
        str(dataset_config.threed_front_dir),
        str(dataset_config.model_info_path),
        str(dataset_config.threed_future_dir),
        str(dataset_config.dump_dir_to_scenes),
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        json_files = json_files,
        filter_fn=lambda s: s)
    print(d)

    '''Read rendering information'''
    scene_render_dir = dataset_config.threed_front_rendering_dir.joinpath('.'.join(args.json_file.split('.')[:-1]))
    cam_K = dataset_config.cam_K

    room_imgs = []
    room_depths = []
    cam_Ts = []
    class_maps = []
    instance_attrs = []
    projected_inst_boxes = []
    for render_path in scene_render_dir.iterdir():
        with h5py.File(render_path) as f:
            colors = np.array(f["colors"])[:,::-1]
            depth = np.array(f["depth"])[:, ::-1]
            depth[depth == dataset_config.infinite_depth] = 0
            cam_T = np.array(f["cam_Ts"])
            class_segmap = np.array(f["class_segmaps"])[:,::-1]
            instance_segmap = np.array(f["instance_segmaps"])[:,::-1]
            instance_attribute_mapping = json.loads(f["instance_attribute_maps"][()])

        ### get scene_name
        scene_json = render_path.parent.name

        #### class mapping
        class_segmap = label_mapping_2D(class_segmap, dataset_config.label_mapping)

        #### get instance info
        inst_marks = set([inst['inst_mark'] for inst in instance_attribute_mapping if
                          inst['inst_mark'] != '' and 'layout' not in inst['inst_mark']])

        inst_info = []
        for inst_mark in inst_marks:
            parts = [part for part in instance_attribute_mapping if part['inst_mark'] == inst_mark]

            # remove background objects.
            category_id = dataset_config.label_mapping[parts[0]['category_id']]
            if category_id == 0:
                continue

            # get 2D masks
            part_indices = [part['idx'] for part in parts]
            inst_mask = np.sum([instance_segmap==idx for idx in part_indices], axis=0, dtype=bool)

            # get 2D bbox
            mask_mat = np.argwhere(inst_mask)
            y_min, x_min = mask_mat.min(axis=0)
            y_max, x_max = mask_mat.max(axis=0)
            bbox = [x_min, y_min, x_max-x_min+1, y_max-y_min+1] # [x,y,width,height]
            if min(bbox[2:]) <= dataset_config.min_bbox_edge_len:
                continue

            inst_dict = {key: parts[0][key] for key in ['inst_mark', 'uid', 'jid', 'room_id', 'location']}
            inst_dict['category_id'] = category_id
            inst_dict['mask'] = inst_mask[y_min:y_max + 1, x_min:x_max + 1]
            inst_dict['bbox2d'] = bbox

            # get 3D bbox
            inst_rm_uid = "_".join([scene_json, inst_dict['room_id']])
            inst_3d_info = parse_inst_from_3dfront(inst_dict, d.rooms, inst_rm_uid)
            inst_dict = {**inst_dict, **inst_3d_info, **{'room_uid': inst_rm_uid}}

            inst_info.append(inst_dict)

        # process cam_T from blender to ours
        cam_T = dataset_config.blender2opengl_cam(cam_T)
        room_imgs.append(colors)
        room_depths.append(depth)
        cam_Ts.append(cam_T)
        class_maps.append(class_segmap)
        instance_attrs.append(inst_info)

        '''Project objects 3D boxes to image planes'''
        projected_box2d_list = project_insts_to_2d(inst_info, cam_K, cam_T)
        projected_inst_boxes.append(projected_box2d_list)

    # get room layout information
    layout_boxes = []
    for rm in d.rooms:
        layout_boxes.append(rm.layout_box)

    viser_2D = VIS_3DFRONT_2D(color_maps=room_imgs, depth_maps=room_depths, inst_info=instance_attrs, cls_maps=class_maps,
                              class_names=dataset_config.label_names, projected_inst_boxes=projected_inst_boxes)

    viser_2D.draw_colors()
    viser_2D.draw_depths()
    viser_2D.draw_cls_maps()
    viser_2D.draw_inst_maps(type=('mask'))
    viser_2D.draw_box2d_from_3d()

    viser = VIS_3DFRONT(rooms=d.rooms, cam_K=cam_K, cam_Ts=cam_Ts, color_maps=room_imgs, depth_maps=room_depths,
                        inst_info=instance_attrs, layout_boxes=layout_boxes,
                        class_names=dataset_config.label_names)
    viser.visualize(type=['pointcloud', 'mesh', 'bbox', 'layout_box', 'cam_pose', 'ori_layout'])
