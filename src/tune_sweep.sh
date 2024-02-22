#!/usr/bin/env sh
#SBATCH --job-name=wandb-sweep
#SBATCH --nodes=1
#SBATCH -p gpu-8
#SBATCH --gres=gpu:a6000:2
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
# export CUDA_VISIBLE_DEVICES=`echo $SLURM_JOB_GPUS | tr ',' ' '`

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi -L

sweep_id=ab742rfe

# srun --exclusive -l wandb agent koffi-anderson/usb_side_channel/${sweep_id}
wandb agent koffi-anderson/usb_side_channel/${sweep_id}
# srun --exclusive --gres=gpu:a100:2 -l wandb agent koffi-anderson/usb_side_channel/${sweep_id}
