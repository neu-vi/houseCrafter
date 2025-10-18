#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:8
#SBATCH --time=48:00:00
#SBATCH --job-name=train_maked_depth
#SBATCH --mem=128
#SBATCH --ntasks=64
#SBATCH --output=myjob.train_maked_depth.out
#SBATCH --error=myjob.train_maked_depth.err

cd ../..
export HF_HOME="/projects/vig/hieu/.cache/huggingface"
export TORCH_HOME="/projects/vig/hieu/.cache/torch"
export WANDB_DATA_DIR="/projects/vig/hieu/.cache/wandb"
export WANDB_DIR="/projects/vig/hieu/.cache/wandb"
export WANDB_CONFIG_DIR="/projects/vig/hieu/.cache/wandb/config"
export WANDB_CACHE_DIR="/projects/vig/hieu/.cache/wandb/artifacts"

accelerate launch \
  --config_file accelerate_mgpu_cfg.yaml \
  --num_processes 1 \
  --main_process_port 29555 \
  6DoF/train_eschernet_explorer.py \
  --cfg \
  "./6DoF/configs/base.yaml" \
  "./6DoF/configs/base_layoutobj_rcn_iodepth.yaml" \
  "./6DoF/configs/3dfront_layoutobj_mask_depth_curated_explorer.yaml" \
  --tracker_project_name eschernet \
  --weights /projects/vig/hieu/escher/logs_eschernet_3dfront_layoutobj_iodepth_1871_scene_maked_depth_nvsfinetune/pipeline-1000
