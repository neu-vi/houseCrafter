import argparse
import os
import subprocess

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument(
        "output_folder",
        nargs="?",
        help="Path to where the data should be saved",
    )
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--layout_folder", type=str, required=True)
    parser.add_argument("--error_folder", type=str, default="./error")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=7000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--save_pose", type=bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser.parse_args()


def run_per_scene(args, scene_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cmd = f" \
blenderproc run \
{os.path.dirname(__file__)}/render_layout.py \
{args.front_folder} \
{args.output_folder} \
--poses_folder '{args.poses_folder}' \
--locations_folder '{args.locations_folder}' \
--layout_folder '{args.layout_folder}' \
--error_folder {args.error_folder} \
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

    os.makedirs(args.output_folder, exist_ok=True)
    scenes = sorted(
        [f.split(".")[0] for f in os.listdir(args.front_folder) if f.endswith(".json")]
    )
    scenes = scenes[args.start : args.end]
    scenes = set(scenes[args.offset :: args.step])
    os.makedirs(args.error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.error_folder)]
    for scene_id in tqdm(scenes):
        if scene_id in error_scenes:
            continue
        run_per_scene(args, scene_id)
    # scene_id = "6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9"
    # if args.locations_folder:
    #     render_scene_from_location(scene_id, args, seed=0)
    # if args.poses_folder:
    #     render_scene_from_pose(scene_id, args, seed=0)

"""
d71

python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 0 \
--gpu_id 0

python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 1 \
--gpu_id 1

python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 2 \
--gpu_id 2

python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 3 \
--gpu_id 3


python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 4 \
--gpu_id 4


python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 5 \
--gpu_id 5


python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 6 \
--gpu_id 6


python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano \
--locations_folder /work/vig/hieu/3dfront_data/locations_100_scene_7 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_pano_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 8 \
--offset 7 \
--gpu_id 7








############### render from random poses on floor

python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_100scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_100 \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_100scenes_random_floor_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--end 100 \
--step 2 \
--gpu_id 0 \
--offset 0 


########### render 500 scenes from random poses on floor
# cpu 1000 scenes
python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_1000scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_1000scenes_random_floor_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--start 100 \
--end 1000 \
--step 38 \
--gpu_id 0 \
--offset 0

# gpu
python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_1000scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_1000scenes_random_floor_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--start 100 \
--end 1000 \
--step 6 \
--gpu_id 5 \
--offset 5

# cpu 2000 scenes
python examples/datasets/front_3d_with_improved_mat/render_layouts.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/hieu/3dfront_data/layout_pcd_2000scenes_random_floor \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/layout_pcd_2000scenes_random_floor_error \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--start 1000 \
--end 2000 \
--step 37 \
--gpu_id 0 \
--offset 0
"""
