import argparse
import os

import numpy as np
from front3d_scene import load_scene, load_labels
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_folder", type=str, help="Path to the 3D front file")
    parser.add_argument("--label_id_mapping_file", type=str)
    parser.add_argument("--model_info_file", type=str)
    parser.add_argument(
        "--future_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument(
        "--future_bbox_folder", type=str, help="Path to the 3D Future Model folder."
    )
    parser.add_argument("--layout_folder", type=str)
    parser.add_argument("--error_folder", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=7000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    return parser.parse_args()


def get_layout(scene_id, args, label_id_mapping, model_id_to_label):
    scene_file = os.path.join(args.front_folder, f"{scene_id}.json")
    house = load_scene(scene_file, args.future_folder, args.future_bbox_folder)
    output_file = os.path.join(args.layout_folder, f"{scene_id}.json")
    house.save_furniture_2d_boxes(label_id_mapping, model_id_to_label, output_file)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.layout_folder, exist_ok=True)
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

    label_id_mapping, model_id_to_label, labelid2name = load_labels(
        args.label_id_mapping_file, args.model_info_file
    )
    for index, scene_id in tqdm(scene_id_dict.items()):
        if os.path.exists(os.path.join(args.layout_folder, f"{scene_id}.json")):
            continue
        if scene_id in error_scenes:
            continue
        try:
            get_layout(scene_id, args, label_id_mapping, model_id_to_label)
        except Exception as e:
            with open(os.path.join(args.error_folder, f"{scene_id}.txt"), "w") as f:
                f.write(str(e))
            print(f"Error in scene {scene_id}")
