import argparse
import os
import subprocess
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("blenderproc_script_path", help="Per rendering script path.")
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument('cc_material_folder', nargs='?', default="resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("output_folder", nargs='?', default="examples/datasets/front_3d_with_improved_mat/renderings",
                        help="Path to where the data should be saved")
    parser.add_argument("--n_views_per_scene", type=int, default=100,
                        help="The number of views to render in each scene.")
    parser.add_argument("--append_to_existing_output", type=bool, default=True,
                        help="If append new renderings to the existing ones.")
    parser.add_argument("--n_processes", type=int, default=1,
                        help="Number of rendering processes to run in parallel. less than available GPUs.")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=480, help="Image width.")
    parser.add_argument("--res_y", type=int, default=360, help="Image height.")
    return parser.parse_args()


def get_folders(args):
    front_folder = Path(args.front_folder)
    future_folder = Path(args.future_folder)
    front_3D_texture_folder = Path(args.front_3D_texture_folder)
    cc_material_folder = Path(args.cc_material_folder)
    output_folder = Path(args.output_folder)
    return front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder

def per_call(process_id, args, front_jsons_by_process):
    # set visible GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(process_id)

    per_render_script_path = args.blenderproc_script_path
    for front_json in front_jsons_by_process[process_id]:
        cmd = ["blenderproc", "run", per_render_script_path, str(front_folder), str(future_folder),
               str(front_3D_texture_folder), str(front_json), str(cc_material_folder), str(output_folder),
               '--n_views_per_scene', str(args.n_views_per_scene),
               '--append_to_existing_output', str(args.append_to_existing_output)]
        print(" ".join(cmd))
        # execute one BlenderProc run
        subprocess.run(" ".join(cmd), env=env, shell=True, check=True)


if __name__ == '__main__':
    '''Parse folders / file paths'''
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)

    front_jsons = [front_json.name for front_json in front_folder.iterdir()]

    '''Pass already generated and failed scenes.'''
    failed_scene_name_file = output_folder.parent.joinpath('failed_scene_names.txt')
    if failed_scene_name_file.is_file():
        with open(failed_scene_name_file, 'r') as file:
            failure_scenes = file.read().splitlines()
    else:
        failure_scenes = []

    filtered_fron_jsons = []
    for json_file in front_jsons:
        scene_name = os.path.splitext(json_file)[0]
        scene_output_folder = output_folder.joinpath(scene_name)
        if scene_output_folder.is_dir():
            existing_n_renderings = len(list(scene_output_folder.iterdir()))
            if existing_n_renderings >= args.n_views_per_scene:
                print('Scene %s is already generated.' % (scene_output_folder.name))
                continue
        if scene_name in failure_scenes:
            print('File in failure log: %s. Continue.' % (scene_name))
            continue
        filtered_fron_jsons.append(json_file)

    front_jsons_by_process = np.array_split(filtered_fron_jsons, args.n_processes)

    p = Pool(processes=args.n_processes)
    p.map(partial(per_call, args=args, front_jsons_by_process=front_jsons_by_process), range(args.n_processes))
    p.close()
    p.join()
