#!/usr/bin/env sh
#SBATCH --job-name=wandb-sweep
#SBATCH --nodes=1
#SBATCH -p gpu-8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --time=12:00:00
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

# srun --exclusive -l wandb agent koffi-anderson/usb_side_channel/${sweep_id}
# srun --exclusive --gres=gpu:a100:2 -l wandb agent koffi-anderson/usb_side_channel/${sweep_id}

# dataset_d1-tsfresh
sweep_id="dpgkapyt"

# # dataset_d1-raw
# sweep_id="h55iq2ps"

# # dataset_d2-raw
# sweep_id="kzkidy6n"



wandb agent koffi-anderson/usb_side_channel/${sweep_id}


# # dataset b tsfresh
# wandb agent --count 20 koffi-anderson/usb_side_channel/9tcucqwm

# # dataset c1 tsfresh
# wandb agent --count 20 koffi-anderson/usb_side_channel/fdo8h8t6

# # dataset d1 tsfresh
# wandb agent --count 20 koffi-anderson/usb_side_channel/c5e8baot

# # dataset d2 tsfresh
# wandb agent --count 20 koffi-anderson/usb_side_channel/olj9d481

# # # dataset a encoder
# wandb agent --count 20 koffi-anderson/usb_side_channel/x2ybywkr

# # # dataset c1 encoder
# wandb agent --count 20 koffi-anderson/usb_side_channel/pbtfqo9m
