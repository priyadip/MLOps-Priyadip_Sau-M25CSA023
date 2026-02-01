#!/bin/bash
#SBATCH --job-name=cifar10_convnext
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=cifar10_convnext_%j.log

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# ================= USER =================
USERNAME="m25csa023"
# ========================================

# -------- SCRATCH DIRECTORIES ----------
export SCRATCH_DIR="/scratch/data/${USERNAME}/Lab2"
export DATA_DIR="${SCRATCH_DIR}/cifar10_data"
export RESULTS_DIR="${SCRATCH_DIR}/cifar10_results"
export MODELS_DIR="${SCRATCH_DIR}/cifar10_models"

mkdir -p $DATA_DIR $RESULTS_DIR $MODELS_DIR

echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Models dir: $MODELS_DIR"

# -------- UNBUFFERED OUTPUT -------------
export PYTHONUNBUFFERED=1

# -------- CONDA SETUP (IMPORTANT) -------
export CONDARC=/scratch/data/${USERNAME}/conda/condarc

module purge
module load anaconda3/2024
source ~/.bashrc
conda activate dlops

echo "Loaded modules:"
module list

# -------- INSTALL REQUIRED PACKAGES --------
echo "=========================================="
echo "Installing required packages..."
echo "=========================================="
pip install thop --quiet
pip install wandb --quiet
pip install scikit-learn --quiet

# -------- WANDB LOGIN (if needed) --------
# Uncomment and set your API key if not already logged in
export WANDB_API_KEY="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"
export WANDB_RUN_NAME="convnextv2_dp_ls_${SLURM_JOB_ID}"
export WANDB_DIR="/scratch/data/${USERNAME}/Lab2/wandb_runs"

# -------- GPU + TORCH CHECK -------------
echo "=========== GPU INFO ==================="
nvidia-smi
python - <<EOF
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
EOF
echo "========================================"

# -------- GO TO WORK DIR ----------------
cd /scratch/data/${USERNAME}/Lab2 || exit 1

echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

echo "=========================================="
echo "Starting CIFAR-10 ConvNeXt-V2 Training"
echo "=========================================="

# -------- RUN TRAINING ----------------
python -u train.py \
    --data-dir $DATA_DIR \
    --results-dir $RESULTS_DIR \
    --models-dir $MODELS_DIR \
    --epochs 60 \
    --batch-size 128 \
    --lr 3e-4 \
    --weight-decay 5e-4 \
    --num-workers 4 \
    --seed 42 \
    --amp \
    --wandb-project "cifar10-convnext-v2"

echo "=========================================="
echo "Training Complete!"
echo "=========================================="

# -------- SHOW RESULTS ----------------
echo "Results saved in: $RESULTS_DIR"
ls -la $RESULTS_DIR

echo "Models saved in: $MODELS_DIR"
ls -la $MODELS_DIR

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="