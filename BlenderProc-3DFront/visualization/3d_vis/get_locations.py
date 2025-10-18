import argparse
import os

import numpy as np
from front3d_scene import load_scene
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_folder", type=str, help="Path to the 3D front file")
    parser.add_argument(
        "--future_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument(
        "--future_bbox_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--dist", type=float)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument("--min_height", type=float, default=1.2)
    parser.add_argument("--max_height", type=float, default=1.7)
    parser.add_argument("--start", type=int, default=0, help="Field of view of camera.")
    parser.add_argument("--error_folder", type=str)
    parser.add_argument(
        "--end", type=int, default=7000, help="Field of view of camera."
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Field of view of camera."
    )
    parser.add_argument("--step", type=int, default=1, help="Field of view of camera.")
    return parser.parse_args()


def get_locations(scene_id, args, seed):
    scene_file = os.path.join(args.front_folder, f"{scene_id}.json")
    house = load_scene(scene_file, args.future_folder, args.future_bbox_folder)
    locations = house.sample_locations(
        dist=args.dist,
        variance=args.variance,
        exclude_furniture=True,
        seed=seed,
        height_range=[args.min_height, args.max_height],
        wall_margin=0.1,
        furniture_margin=0.1,
    )
    location_path = os.path.join(args.locations_folder, f"{scene_id}.npy")
    np.save(location_path, locations)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.locations_folder, exist_ok=True)
    scene_ids = sorted(
        [f.split(".")[0] for f in os.listdir(args.front_folder) if f.endswith(".json")]
    )
    scene_id_dict = {i: scene_id for i, scene_id in enumerate(scene_ids)}
    scene_ids = scene_ids[args.start : args.end]
    scene_ids = set(scene_ids[args.offset :: args.step])
    scene_id_dict = {
        i: scene_id for i, scene_id in scene_id_dict.items() if scene_id in scene_ids
    }
    os.makedirs(args.error_folder, exist_ok=True)
    error_scenes = [x.split(".")[0] for x in os.listdir(args.error_folder)]
    for index, scene_id in tqdm(scene_id_dict.items()):
        if os.path.exists(os.path.join(args.locations_folder, f"{scene_id}.npy")):
            continue
        if scene_id in error_scenes:
            continue
        try:
            get_locations(scene_id, args, seed=index)
        except ValueError as e:
            with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
                f.write(str(e))
            print(f"Error in scene {scene_id}")
