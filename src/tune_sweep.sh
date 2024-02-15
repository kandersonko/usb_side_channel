#!/usr/bin/env sh
#SBATCH --job-name=wandb-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu-8
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=08:00:00
#SBATCH --output=jobs/sweep_%A_%a.stdout
#SBATCH --error=jobs/sweep_%A_%a.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"

hostname
source ~/.bashrc

# activate conda environment
conda activate usb2

python --version
which python

# Assign each task to a specific GPU
# export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

sweep_id=orn54694

# wandb agent koffi-anderson/usb_side_channel/${sweep_id}

srun --exclusive --gres=gpu:a6000:1 -l wandb agent koffi-anderson/usb_side_channel/${sweep_id}
