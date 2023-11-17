#!/usr/bin/env sh
#SBATCH -N 1
#SBATCH -p gpu-volatile-ui
#SBATCH --gres=gpu:rtxa6000:8
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"


hostname
source ~/.bashrc

# activate conda environment
conda activate usb

python --version
which python

nvidia-smi -L

# Execute your script
# python training.py
srun python training.py
