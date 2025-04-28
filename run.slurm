#!/bin/bash
#SBATCH -A trn040
#SBATCH -J finetune_llama3_ds
#SBATCH -o logs/finetune_llama3_ds-%j.o
#SBATCH -e logs/finetune_llama3_ds-%j.e
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH -N 1

# 1) Reset modules & load conda
module reset
module load miniforge3/23.11.0

# 2) Enable `conda activate`
source /sw/odo/miniforge3/23.11.0/etc/profile.d/conda.sh

# 3) Activate your finetuning env (with torch, deepspeed, etc.)
conda activate /gpfs/wolf2/olcf/trn040/world-shared/sajal/env-ft-6.1.3-ro

# 4) Load ROCm if your PyTorch build needs it
module load rocm/6.1.3
export ROCM_HOME=/opt/rocm-6.1.3

# 5) HF offline mode (if pre-cached)
export HF_HUB_OFFLINE=1

# 6) (Optional) proxies / token
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
export HF_TOKEN="YOUR_HF_TOKEN"

# 7) Sanity check
echo "===== ENVIRONMENT CHECK ====="
echo "Python: $(which python)"
python - <<'PYCODE'
import torch, deepspeed
print("Torch:", torch.__version__)
print("DeepSpeed:", deepspeed.__version__)
PYCODE
echo "============================="

# 8) Data preparation
echo ">>> Step 1/2: data prep via sft_llama_ds_own_data.py"
python sft_llama_ds_own_data.py \
  && echo "Data prep succeeded" \
  || { echo "🔴 Data prep failed" ; exit 1; }

# 9) Fine-tuning with DeepSpeed
echo ">>> Step 2/2: training via sft_llama_ds.py"
deepspeed ./sft_llama_ds.py \
  && echo "✅ Training complete" \
  || { echo "🔴 Training failed" ; exit 2; }
