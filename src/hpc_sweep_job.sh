#!/usr/bin/env sh
#SBATCH --job-name=param-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu-volatile
#SBATCH --gres=gpu:a100:2
#SBATCH --time=08:00:00
#SBATCH --output=jobs/sweep_%A_%a.stdout
#SBATCH --error=jobs/sweep_%A_%a.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
# echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

hostname
source ~/.bashrc

# module load cuda/11.8

nvidia-smi -L

# activate conda environment
conda activate usb2

python --version

which python

# fix issues of pytorch with cudnn library
# unset LD_LIBRARY_PATH

# Assign each task to a specific GPU
# export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

sweep_id=sky6l10o

srun --exclusive --gres=gpu:a100:2 -l wandb agent koffi-anderson/usb_side_channel/${sweep_id}
