#!/bin/bash
#SBATCH -A trn040
#SBATCH -J full_finetune_pipeline
#SBATCH -o logs/full_finetune_pipeline-%j.out
#SBATCH -e logs/full_finetune_pipeline-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH -t 12:00:00

# ——— 1) Setup environment —————————————————————
module reset
module load miniforge3/23.11.0
source /sw/odo/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate /gpfs/wolf2/olcf/trn040/world-shared/sajal/env-ft-6.1.3-ro

# ——— 2) Move to project directory —————————————————
cd /gpfs/wolf2/olcf/trn040/scratch/josephagada/project/finetuning

# ——— 3) Sanity check ————————————————————————
echo "HOST:        $(hostname)"
echo "START TIME:  $(date)"
echo "PYTHON:      $(which python)"
echo "GPUS:        $SLURM_GPUS_ON_NODE"
echo "CPUS:        $SLURM_CPUS_ON_NODE"
echo "-------------------------------"

# ——— 4) Step 1: Data & Model Download —————————————
echo ">>> Step 1: sft_llama_ds_own_data.py"
srun python3 sft_llama_ds_own_data.py \
     && echo "Data prep ✅ at $(date)" \
     || { echo "❌ Data prep failed"; exit 1; }

# ——— 5) Step 2: Fine-tuning —————————————————————
echo ">>> Step 2: fine-tuning with sft_llama_ds_own_data.py"
srun deepspeed --num_gpus=8 \
     --master_addr=$MASTER_ADDR \
     --master_port=29500 \
     sft_llama.py \
     --deepspeed ds_config.json \
     && echo "Fine-tuning ✅ at $(date)" \
     || { echo "❌ Fine-tuning failed"; exit 2; }

echo "==== ALL DONE at $(date) ===="
