import blenderproc as bproc
import sys
import argparse
import os
import numpy as np
import json
import blenderproc.python.renderer.RendererUtility as RendererUtility
import quaternion as qt
from tqdm import tqdm
import lmdb
from collections import Counter
import zlib
import cv2
import math
from multiprocessing import Pool
import subprocess
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_folder", help="Path to the 3D front file")
    parser.add_argument("--future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument(
        "--front_3D_texture_folder", 
    )
    parser.add_argument(
        "--cc_material_folder",
        default="resources/cctextures",
    )
    parser.add_argument("--output_image_folder", type=str, default="")
    parser.add_argument("--output_layout_folder", type=str, default="")
    parser.add_argument("--output_depth_folder", type=str, default="")
    
    parser.add_argument("--layout_folder", type=str, required=True)
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--error_folder", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--end", type=int, default=7000,
    )
    
    
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()

def worker(gpu_id, args):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpu_ids = args.gpu_ids.split(',')
    worker_id = gpu_ids.index(str(gpu_id))
    cmd = (f"""
            blenderproc run \
            {os.path.dirname(__file__)}/render.py \
            {args.front_folder} \
            {args.future_folder} \
            {args.front_3D_texture_folder} \
            {args.cc_material_folder} \
            {args.output_folder} \
            --poses_folder {args.pose_folder} \
            --locations_folder {args.locations_folder} \
            --error_folder {args.error_folder} \
            --fov {args.fov} \
            --res_x {args.res_x} \
            --res_y {args.res_y} \
            --start {args.start} \
            --end {args.end} \
            --offset {worker_id} \
            --step {len(gpu_ids)} \
            """)
    print(cmd)
    
if __name__ == '__main__':
    args = parse_args()
    p = Pool(processes=args.num_workers)
    gpu_ids = args.gpu_ids.split(',')
    assert len(gpu_ids) == args.num_workers
    p.map(partial(worker, args=args), args.gpu_ids)
    p.close()
    p.join()
