#!/usr/bin/env sh
#SBATCH --job-name=wandb-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16 # Assign 16 CPUs to each GPU task
#SBATCH -p gpu-volatile-ui
#SBATCH --gres=gpu:rtxa6000:8
#SBATCH --time=08:00:00
#SBATCH --output=jobs/sweep_%A.stdout
#SBATCH --error=jobs/sweep_%A.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"


hostname
source ~/.bashrc

# activate conda environment
conda activate usb

python --version
which python

# nvidia-smi -L

sweep_id=1uga0dfw

# for i in $(seq 0 7); do 
#   CUDA_VISIBLE_DEVICES=$i srun wandb agent koffi-anderson/USB/${sweep_id} &
# done

# Launch the wandb agent for each GPU in a loop
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i srun --exclusive --gres=gpu:1 -n 1 -l wandb agent koffi-anderson/USB/${sweep_id} &
done
wait

# srun --exclusive --gres=gpu:rtxa6000:1 -n 8 -l wandb agent koffi-anderson/USB/${sweep_id}
# srun --exclusive --gres=gpu:1 -n 8 -l wandb agent koffi-anderson/USB/${sweep_id}

# srun --exclusive -n 8 -l wandb agent koffi-anderson/USB/${sweep_id}
