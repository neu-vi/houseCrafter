import argparse
import multiprocessing
import subprocess
import time



def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    start: int = 0,
    end: int = 1500,
    offset: int = 0,
    step: int = 1,
    gpu: int = 0,
) -> None:
    item = queue.get()

    # Perform some operation on the item
    command = (
        # f"export DISPLAY=:0.{gpu} &&"
        # f" GOMP_CPU_AFFINITY='0-47' OMP_NUM_THREADS=48 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE "
        f" python examples/datasets/front_3d_with_improved_mat/renders_obj_depth_only.py"
        f" /work/vig/Datasets/3D-Front/3D-FRONT"
        f" /work/vig/Datasets/3D-Front/3D-FUTURE-model"
        f" /work/vig/Datasets/3D-Front/3D-FRONT-texture"
        f" /work/vig/hieu/BlenderProc-3DFront/resources/cctextures/"
        f" /work/vig/hieu/3dfront_data/obj_depth_1500scenes"
        f" --poses_folder /work/vig/hieu/3dfront_data/obj_poses_all"
        f" --valid_pose_folder /work/vig/hieu/3dfront_data/valid_obj_poses_1500"
        f" --error_folder /work/vig/hieu/3dfront_data/obj_depth_1500scenes_error"
        f" --start {start}"
        f" --end {end}"
        f" --step {step}"
        f" --gpu_id {gpu}"
        f" --offset {offset}"
    )
    print(command)
    subprocess.run(command, shell=True)

    with count.get_lock():
        count.value += 1

    queue.task_done()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=999999)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    n_workers = args.end_offset - args.start_offset
    for gpu, offset in enumerate(range(args.start_offset, args.end_offset)):
        gpu = 0 if args.gpu == "0" else gpu
        process = multiprocessing.Process(
            target=worker,
            args=(
                queue,
                count,
                args.start,
                args.end,
                offset,
                args.step,
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

python examples/datasets/front_3d_with_improved_mat/launch2.py --start 0 --end 1500 --start_offset 0 --end_offset 7 --step 7 --gpu 0
"""