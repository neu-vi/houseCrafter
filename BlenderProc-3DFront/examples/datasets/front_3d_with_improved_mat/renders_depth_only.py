import argparse
import os
import subprocess

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument(
        "front_3D_texture_folder", help="Path to the 3D FRONT texture folder."
    )
    parser.add_argument(
        "cc_material_folder",
        nargs="?",
        default="resources/cctextures",
        help="Path to CCTextures folder, see the /scripts for the download script.",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default="examples/datasets/front_3d_with_improved_mat/renderings",
        help="Path to where the data should be saved",
    )
    # parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--error_folder", type=str, default="./error")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    parser.add_argument("--start", type=int, default=0, help="Field of view of camera.")
    parser.add_argument(
        "--end", type=int, default=7000, help="Field of view of camera."
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Field of view of camera."
    )
    parser.add_argument("--step", type=int, default=1, help="Field of view of camera.")
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser.parse_args()


def run_per_scene(args, scene_id, seed):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cmd = f" \
blenderproc run \
{os.path.dirname(__file__)}/render_depth_only.py \
{args.front_folder} \
{args.future_folder} \
{args.front_3D_texture_folder} \
{args.cc_material_folder} \
{args.output_folder} \
--poses_folder '{args.poses_folder}' \
--error_folder {args.error_folder} \
--cpu_threads {args.cpu_threads} \
--fov {args.fov} \
--res_x {args.res_x} \
--res_y {args.res_y} \
--scene_id {scene_id} \
--seed {seed} \
"
    print(cmd)
    subprocess.call(cmd, env=env, shell=True)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    print("test")
    scenes = sorted(
        [f.split(".")[0] for f in os.listdir(args.front_folder) if f.endswith(".json")]
    )
    scene_id_dict = {i: scene_id for i, scene_id in enumerate(scenes)}

    scenes = scenes[args.start : args.end]
    scenes = set(scenes[args.offset :: args.step])
    scene_id_dict = {
        i: scene_id for i, scene_id in scene_id_dict.items() if scene_id in scenes
    }
    os.makedirs(args.error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.error_folder)]
    for index, scene_id in tqdm(scene_id_dict.items()):
        if scene_id in error_scenes:
            continue
        run_per_scene(args, scene_id, seed=index)
    # scene_id = "6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9"
    # if args.locations_folder:
    #     render_scene_from_location(scene_id, args, seed=0)
    # if args.poses_folder:
    #     render_scene_from_pose(scene_id, args, seed=0)

"""
d71

python examples/datasets/front_3d_with_improved_mat/renders_depth_only.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/work/vig/hieu/3dfront_data/images_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/images_100scenes_pano_error \
--end 100 \
--offset 0 \
--step 8 \
--gpu_id 0




############### render from random poses on floor

d65 first 100
python examples/datasets/front_3d_with_improved_mat/renders_depth_only.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/scratch/nguyen.trungh/3d_front_depth/depth_100scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /scratch/nguyen.trungh/3d_front_depth/depth_100scenes_random_floor_error \
--cpu_threads 8 \
--start 0 \
--end 100 \
--step 1 \
--gpu_id 0 \
--offset 0 \

d71 1000
python examples/datasets/front_3d_with_improved_mat/renders_depth_only.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/scratch/nguyen.trungh/3d_front_depth/depth_1000scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /scratch/nguyen.trungh/3d_front_depth/depth_1000scenes_random_floor_error \
--cpu_threads 8 \
--start 100 \
--end 1000 \
--step 6 \
--offset 0 \
--gpu_id 1

####### run gpu 2000 d70
python examples/datasets/front_3d_with_improved_mat/renders_depth_only.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/scratch/nguyen.trungh/3d_front_depth/depth_2000scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /scratch/nguyen.trungh/3d_front_depth/depth_2000scenes_random_floor_error \
--cpu_threads 8 \
--start 1000 \
--end 2000 \
--step 8 \
--offset 7 \
--gpu_id 7


############### run gpu 3000 d65 d66 ########## 
python examples/datasets/front_3d_with_improved_mat/renders_depth_only.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/scratch/nguyen.trungh/3d_front_depth/depth_3000scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/depth_3000scenes_random_floor_error \
--cpu_threads 8 \
--start 2000 \
--end 3000 \
--step 15 \
--gpu_id 7 \
--offset 14


d70 1500 scene obj pose
python examples/datasets/front_3d_with_improved_mat/renders_depth_only.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/BlenderProc-3DFront/resources/cctextures/ \
/work/vig/hieu/3dfront_data/obj_depth_1000scenes \
--poses_folder /work/vig/hieu/3dfront_data/obj_poses_all \
--valid_pose_folder /work/vig/hieu/3dfront_data/valid_obj_poses_1000 \
--image_error_folder /work/vig/hieu/3dfront_data/obj_images_1000scenes_error \
--layout_error_folder /work/vig/hieu/3dfront_data/obj_layout_pcd_1000scenes_error \
--start 0 \
--end 1500 \
--step 65 \
--gpu_id 0 \
--offset 0
"""
