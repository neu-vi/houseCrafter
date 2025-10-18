#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:7
#SBATCH --time=48:00:00
#SBATCH --job-name=train_newobj
#SBATCH --mem=128
#SBATCH --ntasks=64
#SBATCH --output=myjob.train_newobj.out
#SBATCH --error=myjob.train_newobj.err

cd ../..
accelerate launch \
  --config_file accelerate_mgpu_cfg.yaml \
  --num_processes 1 \
  --main_process_port 29555 \
  6DoF/train_eschernet.py \
  --cfg \
  "./6DoF/configs/base.yaml" \
  "./6DoF/configs/base_layoutobj_rcn_iodepth.yaml" \
  "./6DoF/configs/3dfront_layoutobj_novelcontent.yaml" \
  --tracker_project_name eschernet \
  --weights /work/vig/hieu/escher/logs_eschernet_3dfront_layoutobj_iodepth_1871_scene_3m_final_rerender5m/pipeline-10000
