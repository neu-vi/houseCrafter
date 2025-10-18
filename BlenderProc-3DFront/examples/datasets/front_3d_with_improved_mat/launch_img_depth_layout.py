import argparse
import multiprocessing
import os
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_folder", help="Path to the 3D front file")
    parser.add_argument("--future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("--front_3D_texture_folder")
    parser.add_argument("--cc_material_folder")
    parser.add_argument("--output_image_folder", type=str, default="")
    parser.add_argument("--output_layout_folder", type=str, default="")
    parser.add_argument("--output_depth_folder", type=str, default="")

    parser.add_argument("--poses_folder", type=str, default="")
    parser.add_argument("--layout_folder", type=str, required=True)
    parser.add_argument("--locations_folder", type=str, default="")
    parser.add_argument("--locations_graph_folder", type=str, default="")
    parser.add_argument("--image_error_folder", type=str, required=True)
    parser.add_argument("--depth_error_folder", type=str, required=True)
    parser.add_argument("--layout_error_folder", type=str, required=True)

    parser.add_argument("--thread_per_worker", type=int, default=8)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--n_heading", type=int, default=5)
    parser.add_argument("--elevation", type=float, default=40)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=999999)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--tasks", type=str, default="img,layout,depth")

    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=512, help="Image width.")
    parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()


def worker(
    args,
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    offset: int = 0,
    gpu: int = 0,
) -> None:
    item = queue.get()
    cmd = f"""
        python \
        {os.path.dirname(__file__)}/render_img_depth_layout.py \
        {args.front_folder} \
        {args.future_folder} \
        {args.front_3D_texture_folder} \
        {args.cc_material_folder} \
        {args.output_image_folder} \
        {args.output_layout_folder} \
        {args.output_depth_folder} \
        --poses_folder '{args.poses_folder}' \
        --layout_folder {args.layout_folder} \
        --locations_folder '{args.locations_folder}' \
        --locations_graph_folder '{args.locations_graph_folder}' \
        --image_error_folder {args.image_error_folder} \
        --depth_error_folder {args.depth_error_folder} \
        --layout_error_folder {args.layout_error_folder} \
        --cpu_threads {args.thread_per_worker} \
        --n_heading {args.n_heading} \
        --elevation {args.elevation} \
        --fov {args.fov} \
        --res_x {args.res_x} \
        --res_y {args.res_y} \
        --start {args.start} \
        --end {args.end} \
        --offset {offset} \
        --step {args.step} \
        --gpu_id {gpu} \
        --tasks {args.tasks} \
    """
    print(cmd)
    subprocess.run(cmd, shell=True)

    with count.get_lock():
        count.value += 1

    queue.task_done()

if __name__ == "__main__":
    args = parse_args()
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    n_workers = args.end_offset - args.start_offset
    for gpu, offset in enumerate(range(args.start_offset, args.end_offset)):
        gpu = gpu if args.gpu else 0
        process = multiprocessing.Process(
            target=worker,
            args=(
                args,
                queue,
                count,
                offset,
                gpu,
            ),
        )
        process.daemon = True
        process.start()

    for item in range(args.start_offset, args.end_offset):
        queue.put(item)

    # update the wandb count
    while True:
        time.sleep(1000)
        print(
            {
                "count": count.value,
                "total": n_workers,
                "progress": count.value / n_workers,
            }
        )
        if count.value == n_workers:
            break

    # Wait for all tasks to be completed
    queue.join()

    print("Finished", count.value)

"""
cpu 3000-4000
python examples/datasets/front_3d_with_improved_mat/launch_img_depth_layout.py \
--front_folder /work/vig/Datasets/3D-Front/3D-FRONT \
--future_folder /work/vig/Datasets/3D-Front/3D-FUTURE-model \
--front_3D_texture_folder /work/vig/Datasets/3D-Front/3D-FRONT-texture \
--cc_material_folder /work/vig/hieu/BlenderProc-3DFront/resources/cctextures \
--output_image_folder /work/vig/hieu/3dfront_data/images_4000scenes_graphpano \
--output_layout_folder /work/vig/hieu/3dfront_data/layout_pcd_4000scenes_graphpano \
--output_depth_folder /work/vig/hieu/3dfront_data/depths_4000scenes_graphpano \
--layout_folder /work/vig/hieu/3dfront_data/layouts \
--locations_graph_folder /work/vig/hieu/3dfront_data/graph_pano_train \
--image_error_folder /work/vig/hieu/3dfront_data/images_4000scenes_graphpano_error \
--depth_error_folder /work/vig/hieu/3dfront_data/depths_4000scenes_graphpano_error \
--layout_error_folder /work/vig/hieu/3dfront_data/layout_pcd_4000scenes_graphpano_error \
--end 4000 \
--start 3000 \
--step 22 \
--start_offset 16 \
--end_offset 22
"""