import argparse
import os
import subprocess

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder")
    parser.add_argument("cc_material_folder", nargs="?", default="resources/cctextures")
    parser.add_argument("output_image_folder", nargs="?")
    parser.add_argument("output_layout_folder", nargs="?")

    parser.add_argument("--layout_folder", type=str, required=True)
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--valid_pose_folder", type=str, required=True)
    parser.add_argument("--poses_per_obj", type=int, default=20)
    parser.add_argument("--visible_thr", type=int, default=1000)
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--image_error_folder", type=str, required=True)
    parser.add_argument("--layout_error_folder", type=str, required=True)
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    parser.add_argument("--start", type=int, default=0, help="Field of view of camera.")
    parser.add_argument("--depth", type=str, default="1")
    parser.add_argument("--end", type=int, default=7000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_pose", type=bool, default=False)
    parser.add_argument("--tasks", type=str, default="img,layout")

    return parser.parse_args()


def run_per_scene(args, scene_id, seed):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # render images
    if "img" in args.tasks:
        cmd = f" \
blenderproc run \
{os.path.dirname(__file__)}/render_obj.py \
{args.front_folder} \
{args.future_folder} \
{args.front_3D_texture_folder} \
{args.cc_material_folder} \
{args.output_image_folder} \
--poses_folder '{args.poses_folder}' \
--valid_pose_folder '{args.valid_pose_folder}' \
--locations_folder '{args.locations_folder}' \
--error_folder {args.image_error_folder} \
--visible_thr {args.visible_thr} \
--poses_per_obj {args.poses_per_obj} \
--depth {args.depth} \
--fov {args.fov} \
--res_x {args.res_x} \
--res_y {args.res_y} \
--scene_id {scene_id} \
--seed {seed} \
"
        print(cmd)
        subprocess.call(cmd, env=env, shell=True)
    if "layout" in args.tasks:
        # render layouts
        cmd = f" \
blenderproc run \
{os.path.dirname(__file__)}/render_obj_layout.py \
{args.front_folder} \
{args.output_layout_folder} \
--poses_folder '{args.poses_folder}' \
--valid_pose_folder '{args.valid_pose_folder}' \
--locations_folder '{args.locations_folder}' \
--layout_folder '{args.layout_folder}' \
--error_folder {args.layout_error_folder} \
--fov {args.fov} \
--res_x {args.res_x} \
--res_y {args.res_y} \
--scene_id {scene_id} \
--save_pose {args.save_pose} \
"
        print(cmd)
        subprocess.call(cmd, env=env, shell=True)


if __name__ == "__main__":
    args = parse_args()
    assert args.locations_folder or args.poses_folder, "Need locations or poses folder"
    assert not (
        args.locations_folder and args.poses_folder
    ), "Only one of locations or poses folder should be provided"

    os.makedirs(args.output_image_folder, exist_ok=True)
    os.makedirs(args.output_layout_folder, exist_ok=True)
    scenes = sorted(
        [f.split(".")[0] for f in os.listdir(args.front_folder) if f.endswith(".json")]
    )
    scene_id_dict = {i: scene_id for i, scene_id in enumerate(scenes)}

    scenes = scenes[args.start : args.end]
    scenes = set(scenes[args.offset :: args.step])
    scene_id_dict = {
        i: scene_id for i, scene_id in scene_id_dict.items() if scene_id in scenes
    }
    # scene_id_dict = {0: "00ecd5d3-d369-459f-8300-38fc159823dc"} ###############
    os.makedirs(args.image_error_folder, exist_ok=True)
    os.makedirs(args.layout_error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.image_error_folder)]
    error_scenes += [x.split(".")[0] for x in os.listdir(args.layout_error_folder)]
    error_scenes = set(error_scenes)
    for index, scene_id in tqdm(scene_id_dict.items()):
        if scene_id in error_scenes:
            continue
        run_per_scene(args, scene_id, seed=index)

"""
debug local
python examples/datasets/front_3d_with_improved_mat/renders_obj_img_layout.py \
examples/datasets/front_3d_with_improved_mat/3D-FRONT \
examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model \
examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture \
resources/cctextures/ \
/media/hieu/T7/3dfront_render/debug_obj \
/media/hieu/T7/3dfront_render/layout_debug_obj \
--poses_folder /media/hieu/T7/3d-front/obj_poses_100 \
--layout_folder /media/hieu/T7/3d-front/layouts \
--image_error_folder /media/hieu/T7/3dfront_render/debug_obj_error \
--layout_error_folder /media/hieu/T7/3dfront_render/debug_layout_obj_error \
--valid_pose_folder /media/hieu/T7/3d-front/valid_obj_poses_100 \
--start 0 \
--end 100 \
--tasks layout \
--step  \
--offset 5 \
--gpu_id 5

d70 1500
python examples/datasets/front_3d_with_improved_mat/renders_obj_img_layout.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/work/vig/hieu/3dfront_data/obj_images_1500scenes \
/work/vig/hieu/3dfront_data/obj_layout_pcd_1500scenes \
--poses_folder /work/vig/hieu/3dfront_data/obj_poses_all \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--valid_pose_folder /work/vig/hieu/3dfront_data/valid_obj_poses_1500 \
--image_error_folder /work/vig/hieu/3dfront_data/obj_images_1500scenes_error \
--layout_error_folder /work/vig/hieu/3dfront_data/obj_layout_pcd_1500scenes_error \
--depth 0 \
--start 0 \
--end 1500 \
--step 7 \
--gpu_id 6 \
--offset 6



"""
