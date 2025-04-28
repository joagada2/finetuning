#!/bin/bash
#SBATCH -A trn040
#SBATCH -J full_finetune_pipeline
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH -p batch

# 1) Clean slate
module reset

# 2) Load conda & ROCm toolchain
module load miniforge3/23.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /gpfs/wolf2/olcf/trn040/world-shared/sajal/env-ft-6.1.3-ro

module load rocm/6.1.3
module load gcc/12.2.0

# 3) Offline HF
export HF_HUB_OFFLINE=1

# 4) MPI/torchrun details (optional, for multi-node)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# 5) Run data prep
echo ">>> STEP 1: preparing data"
srun --ntasks=1 python sft_llama_ds_own_data.py \
     --output_dir ./finetune_data \
  || { echo "DATA PREP FAILED"; exit 1; }

# 6) Build hostfile (if you really need deepspeed multi-node)
# scontrol show hostnames $SLURM_NODELIST > host_list.$SLURM_JOB_ID
# paste -d' ' host_list.$SLURM_JOB_ID > hosts.$SLURM_JOB_ID  # simple 1-node, 1-line

# 7) Run fine-tuning
echo ">>> STEP 2: launching deepspeed"
srun --ntasks=1 deepspeed \
      --num_nodes 1 \
      --num_gpus 8 \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      sft_llama_ds.py \
  || { echo "FINE-TUNE FAILED"; exit 1; }

echo "✅ ALL DONE"
