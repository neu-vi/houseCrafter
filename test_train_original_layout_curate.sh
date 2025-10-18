#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:6
#SBATCH --time=48:00:00
#SBATCH --job-name=train_curate
#SBATCH --mem=128
#SBATCH --ntasks=64
#SBATCH --output=myjob.train_curate.out
#SBATCH --error=myjob.train_curate.err

accelerate launch \
  --config_file accelerate_mgpu_cfg.yaml \
  --num_processes 1 \
  --main_process_port 29555 \
  src/train_eschernet_explorer.py \
  --cfg \
  "./src/configs/base.yaml" \
  "./src/configs/base_layout_rcn_iodepth_v.yaml" \
  "./src/configs/3dfront_layout_rand_curate_explorer.yaml" \
  --tracker_project_name eschernet \
  --weights /projects/vig/hieu/escher/logs_eschernet_3dfront_layout_rand_curate/pipeline-2500

