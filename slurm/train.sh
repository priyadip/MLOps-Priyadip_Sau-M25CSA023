#!/bin/bash
#SBATCH --job-name=dlops_bert_train
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=dlops_train_%j.log

echo "=========================================="
echo "SLURM Job ID  : $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time     : $(date)"
echo "=========================================="

# ================= USER CONFIG =================
USERNAME=""
PROJECT_DIR="/scratch/data/${USERNAME}/dlops-assignment3"
# ===============================================

# -------- SCRATCH DIRECTORIES ------------------
export SCRATCH_DIR="/scratch/data/${USERNAME}"
mkdir -p "${PROJECT_DIR}/data" "${PROJECT_DIR}/results" "${PROJECT_DIR}/saved_model"

# -------- CONDA SETUP -------------------------
export CONDARC=/scratch/data/${USERNAME}/conda/condarc

module purge
module load anaconda3/2024
source ~/.bashrc
conda activate /scratch/data/m25csa023/conda/envs/dlops

echo "Loaded modules:"
module list

# -------- GPU + TORCH CHECK --------------------
echo "=========== GPU INFO ==================="
nvidia-smi
python - <<EOF
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Memory:", round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), "GB")
EOF
echo "========================================"

# -------- RUN TRAINING -------------------------
cd "${PROJECT_DIR}" || exit 1
echo "Working directory: $(pwd)"
echo "Starting training..."

python scripts/train.py \
    --epochs 3 \
    --batch_size 10 \
    --lr 5e-5 \
    --output_dir ./saved_model \
    --data_cache ./data/genre_reviews.pkl

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
