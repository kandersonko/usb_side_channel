#!/usr/bin/env sh
#SBATCH --job-name=param-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu-long
#SBATCH --gres=gpu:1080ti:2
#SBATCH --time=12:00:00
#SBATCH --array=1
#SBATCH --output=jobs/sweep_%A_%a.stdout
#SBATCH --error=jobs/sweep_%A_%a.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

hostname
source ~/.bashrc

module load cuda/11.8

nvidia-smi -L

# activate conda environment
conda activate usb

python --version

which python

# fix issues of pytorch with cudnn library
unset LD_LIBRARY_PATH

# Assign each task to a specific GPU
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

sweep_id=2du09tko

srun --exclusive --gres=gpu:1080ti:2 -l wandb agent koffi-anderson/USB/${sweep_id}
