#!/usr/bin/env sh
#SBATCH --job-name=features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu-long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs/features_%A_%a.stdout
#SBATCH --error=jobs/features_%A_%a.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"

hostname

lscpu

source ~/.bashrc

# activate conda environment
conda activate usb2

python --version
which python

# run feature extraction
# srun ./run_extraction.sh
srun --exclusive --gres=gpu:a6000:1 -l ./run_extraction.sh

# run feature engineering
srun ./run_tsfresh.sh



wait
