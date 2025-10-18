import argparse
import os
import subprocess
import json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument(
        "front_3D_texture_folder", help="Path to the 3D FRONT texture folder."
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default="examples/datasets/front_3d_with_improved_mat/renderings",
        help="Path to where the data should be saved",
    )
    parser.add_argument("--included_scenes", type=str, default="")
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--poses_folder", type=str, default="")
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
{os.path.dirname(__file__)}/render_semseg.py \
{args.front_folder} \
{args.future_folder} \
{args.front_3D_texture_folder} \
{args.output_folder} \
--poses_folder '{args.poses_folder}' \
--locations_folder '{args.locations_folder}' \
--error_folder {args.error_folder} \
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
    assert args.locations_folder or args.poses_folder, "Need locations or poses folder"
    assert not (
        args.locations_folder and args.poses_folder
    ), "Only one of locations or poses folder should be provided"

    os.makedirs(args.output_folder, exist_ok=True)
    scenes = sorted(
        [f.split(".")[0] for f in os.listdir(args.front_folder) if f.endswith(".json")]
    )
    scene_id_dict = {i: scene_id for i, scene_id in enumerate(scenes)}

    scenes = scenes[args.start : args.end]
    if args.included_scenes:
        included_scenes = set(json.load(open(args.included_scenes, "r")))
        scenes = set(scenes) & included_scenes
        scenes = sorted(list(scenes))
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


"""
0-100
python examples/datasets/front_3d_with_improved_mat/render_semsegs.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/3dfront_data/seg_100scenes_random_floor \
--included_scenes /work/vig/hieu/3dfront_data/all_seg_scenes_1711_thr_5scene_2obj.json \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/seg_100scenes_random_floor_error \
--start 0 \
--end 100 \
--step 1 \
--offset 0 \
--gpu_id 0

100-1000
python examples/datasets/front_3d_with_improved_mat/render_semsegs.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/3dfront_data/seg_1000scenes_random_floor \
--included_scenes /work/vig/hieu/3dfront_data/all_seg_scenes_1711_thr_5scene_2obj.json \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/seg_1000scenes_random_floor_error \
--start 100 \
--end 1000 \
--step 9 \
--gpu_id 0 \
--offset 0

1000-2000
python examples/datasets/front_3d_with_improved_mat/render_semsegs.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/3dfront_data/seg_2000scenes_random_floor \
--included_scenes /work/vig/hieu/3dfront_data/all_seg_scenes_1711_thr_5scene_2obj.json \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/seg_2000scenes_random_floor_error \
--start 1000 \
--end 2000 \
--step 10 \
--gpu_id 0 \
--offset 0


2000-3000
python examples/datasets/front_3d_with_improved_mat/render_semsegs.py \
/work/vig/Datasets/3D-Front/3D-FRONT \
/work/vig/Datasets/3D-Front/3D-FUTURE-model \
/work/vig/Datasets/3D-Front/3D-FRONT-texture \
/work/vig/hieu/3dfront_data/seg_3000scenes_random_floor \
--included_scenes /work/vig/hieu/3dfront_data/all_seg_scenes_1711_thr_5scene_2obj.json \
--poses_folder /work/vig/hieu/3dfront_data/poses_all \
--error_folder /work/vig/hieu/3dfront_data/seg_3000scenes_random_floor_error \
--start 2000 \
--end 3000 \
--step 12 \
--gpu_id 0 \
--offset 0
"""
