#!/usr/bin/env sh
#SBATCH --job-name=wandb-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu-volatile-ui
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=jobs/sweep_%A_%a.stdout
#SBATCH --error=jobs/sweep_%A_%a.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

hostname
source ~/.bashrc

# activate conda environment
conda activate usb

python --version
which python

# Assign each task to a specific GPU
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

sweep_id=b9jqdanv
# wandb agent koffi-anderson/USB/${sweep_id}

srun --exclusive --gres=gpu:rtxa6000:1 -l wandb agent koffi-anderson/USB/${sweep_id}
