#!/usr/bin/env sh
#SBATCH -N 1
#SBATCH -p gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/sweep_%A_%a.stdout
#SBATCH --error=jobs/sweep_%A_%a.stderr

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

module load cuda/11.8

nvidia-smi -L

# fix issues of pytorch with cudnn library
unset LD_LIBRARY_PATH

# Execute your script
# python train.py
srun python train.py --min_epochs=10 --max_epochs=500 --batch_size=512 --target_label=class --num_classes=2
