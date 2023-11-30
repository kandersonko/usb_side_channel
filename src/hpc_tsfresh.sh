#!/usr/bin/env sh
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH -p short
#SBATCH --output=jobs/tsfresh_%A.stdout
#SBATCH --error=jobs/tsfresh_%A.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"


hostname

# source /usr/modules/init/bash
source ~/.bashrc

# activate conda environment
conda activate usb

python --version
which python

# module load cuda/11.8

# nvidia-smi -L

# fix issues of pytorch with cudnn library
unset LD_LIBRARY_PATH

# Execute your script
# srun python tsfresh_feature_engineering.py
python tsfresh_feature_engineering.py
