#!/usr/bin/env sh
#SBATCH --job-name=features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=long
#SBATCH --time=08:00:00
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

# run feature engineering
srun ./run_tsfresh.sh

# run feature extraction
srun ./run_extraction.sh


wait
