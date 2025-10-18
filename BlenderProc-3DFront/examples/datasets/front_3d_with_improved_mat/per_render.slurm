#!/bin/sh
#SBATCH --job-name=3dfront_render           # Job name
#SBATCH --output=./slurm_jobs/job_%j.log           # Standard output and error log
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=60gb                     # Job memory request
#SBATCH --gpus=1                       # Job GPUs request
##SBATCH --nodelist=seti
##SBATCH --exclude=lothlann             # Exclude nodes
#SBATCH --cpus-per-task=12
##SBATCH --mail-user=yinyu.nie@tum.de
#SBATCH --qos=normal
#SBATCH --partition=submit

# Default output information
date;hostname;pwd
echo "Job Name = $SLURM_JOB_NAME"

# Your code
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/rhome/ynie/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/rhome/ynie/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/rhome/ynie/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/rhome/ynie/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate blenderproc
blenderproc run examples/datasets/front_3d_with_improved_mat/render_dataset_improved_mat.py ./examples/datasets/front_3d_with_improved_mat/3D-FRONT ./examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model ./examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture 6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json ./resources/cctextures/  examples/datasets/front_3d_with_improved_mat/renderings