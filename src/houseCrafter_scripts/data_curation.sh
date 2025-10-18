#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=curate_data
#SBATCH --mem=32
#SBATCH --ntasks=8
#SBATCH --output=myjob.curate_data.out
#SBATCH --error=myjob.curate_data.err

cd ../..
python ./6DoF/inspect_graph.py\
  --config_file accelerate_mgpu_cfg.yaml \
  --num_processes 1 \
  --cfg \
  "./6DoF/configs/base.yaml" \
  "./6DoF/configs/base_layoutobj_rcn_iodepth.yaml" \
  "./6DoF/configs/3dfront_layoutobj_mask_depth.yaml" \
  --tracker_project_name eschernet \
