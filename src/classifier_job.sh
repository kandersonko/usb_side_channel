#!/usr/bin/env sh
#SBATCH --job-name=classifier-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -p gpu-8
#SBATCH --gres=gpu:a6000:2
#SBATCH --output=jobs/cls_%A_%a.stdout
#SBATCH --error=jobs/cls_%A_%a.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"

hostname
source ~/.bashrc

# module load cuda/11.8

nvidia-smi -L

# activate conda environment
conda activate usb2

python --version

which python

# fix issues of pytorch with cudnn library
unset LD_LIBRARY_PATH

srun python classifier.py --log --method=tsfresh --dataset=dataset_a --target_label=category --classifier=lstm --task=identification --batch_size=32
