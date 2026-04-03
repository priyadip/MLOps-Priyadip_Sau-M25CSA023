# DLOps Assignment 5 (LoRA and IBM ART)

This repository contains the full implementation for Assignment-5 using Python and PyTorch, with all code in `.py` files and using a Docker container.

- Name: Priyadip Sau
- Roll Number: M25CSA023
- Branch: `Assignment 5`
- Report PDF: `<M25CSA023_Priyadip_Sau_Ass5.pdf>`

## Links 

- WandB project/run links:
  - Q1: [WandB Q1](https://wandb.ai/saupriyadip571-indian-institute-of-technology-jodhpur/DLOps-Ass5-Q1)
  - Q1 Optuna: [WandB Optuna](https://wandb.ai/saupriyadip571-indian-institute-of-technology-jodhpur/DLOps-Ass5-Q1-Optuna)
  - Q2: [WandB Q2](https://wandb.ai/saupriyadip571-indian-institute-of-technology-jodhpur/DLOps-Ass5-Q2)

- HuggingFace model link (best Q1 model): [HuggingFace ViT-LoRA](https://huggingface.co/priyadip/vit-lora-cifar100)
- Code: [GitHub Assignment-5](https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/tree/Assignment-5)


## Repository Structure

```text
MLOps-Priyadip_Sau-M25CSA023/
├── q1_train.py
├── q1_optuna.py
├── q1_test.py
├── q2_fgsm.py
├── q2_detector.py
├── utils.py
├── requirements.txt
├── Dockerfile
├── outputs_q1/
│   ├── checkpoints/
│   ├── plots/
│   └── results/
└── outputs_q2/
    ├── checkpoints/
    ├── plots/
    └── results/
```

## Environment Setup

### 1) Clone Assignment-5 branch

```bash
git clone -b Assignment-5 https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023.git
cd MLOps-Priyadip_Sau-M25CSA023
```

### 2) Docker (Recommended: Pull Prebuilt Image)

Use the prebuilt image from Docker Hub for reproducible setup without local rebuild.

```bash
sudo docker pull priyadipsau/dlops-ass5:assignment-5
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  --shm-size=8g \
  priyadipsau/dlops-ass5:assignment-5
```

### 3) Docker (Fallback: Build Locally)

If Docker Hub is unavailable or you need to rebuild from source:

```bash
sudo docker build -t dlops-ass5 .
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  --shm-size=8g \
  dlops-ass5
```

Inside container:

```bash
cd /workspace
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
wandb login
huggingface-cli login
```

## How To Run

## Q1: ViT-S on CIFAR-100 (Baseline + LoRA + Optuna)

### 1) Baseline (No LoRA)

```bash
python q1_train.py --mode baseline --epochs 10 --output_dir outputs_q1
```

### 2) Single LoRA Experiment

```bash
python q1_train.py --mode lora --rank 4 --alpha 8 --lora_dropout 0.1 --epochs 10 --output_dir outputs_q1
```

### 3) All Required LoRA Combinations

Ranks: 2, 4, 8  |  Alpha: 2, 4, 8  |  Dropout: 0.1

```bash
python q1_train.py --mode all --epochs 10 --output_dir outputs_q1
```

### 4) Optuna Search (LoRA Hyperparameters)

```bash
python q1_optuna.py --n_trials 20 --epochs 10 --output_dir outputs_q1
```

### 5) Test Best Q1 Checkpoint

```bash
python q1_test.py \
  --checkpoint outputs_q1/checkpoints/LoRA_r8_a8_d0.1_best.pt \
  --output_dir outputs_q1
```

### 6) Optional Experiment (LoRA + Partial Unfreezing)

Best config with partially trainable backbone (last 2 ViT blocks trainable + LoRA + trainable head):

```bash
python q1_train.py --mode lora_partial --rank 8 --alpha 8 --lora_dropout 0.1 \
  --partial_unfreeze_last_n 2 --epochs 10 --output_dir outputs_q1
```

This saves results using experiment name pattern:
- `LoRA_r8_a8_d0.1_partial2`

## Q2: Adversarial Robustness with IBM ART

### (i) FGSM: Scratch vs IBM ART (CIFAR-10, ResNet18 from scratch)

Train only:

```bash
python q2_fgsm.py --mode train --epochs 30 --output_dir outputs_q2
```

Attack comparison only (requires checkpoint):

```bash
python q2_fgsm.py --mode attack \
  --checkpoint outputs_q2/checkpoints/resnet18_cifar10_best.pt \
  --output_dir outputs_q2
```

Full pipeline:

```bash
python q2_fgsm.py --mode full --epochs 30 --output_dir outputs_q2
```

### (ii) Adversarial Detector (PGD and BIM, ResNet34)

```bash
python q2_detector.py --mode full --epochs_base 30 --epochs_detector 20 --output_dir outputs_q2
```

or detector-only mode:

```bash
python q2_detector.py --mode detector \
  --checkpoint outputs_q2/checkpoints/resnet18_cifar10_best.pt \
  --output_dir outputs_q2
```

## Q1 Results Summary (From outputs_q1/results/all_results_summary.json)

| Experiment | LoRA | Rank | Alpha | Dropout | Test Accuracy (%) | Trainable Params |
|---|---|---:|---:|---:|---:|---:|
| Baseline_NoLoRA | No | 0 | 0 | 0.1 | 80.96 | 38,500 |
| LoRA_r2_a2_d0.1 | Yes | 2 | 2 | 0.1 | 89.12 | 75,364 |
| LoRA_r2_a4_d0.1 | Yes | 2 | 4 | 0.1 | 89.04 | 75,364 |
| LoRA_r2_a8_d0.1 | Yes | 2 | 8 | 0.1 | 89.43 | 75,364 |
| LoRA_r4_a2_d0.1 | Yes | 4 | 2 | 0.1 | 89.40 | 112,228 |
| LoRA_r4_a4_d0.1 | Yes | 4 | 4 | 0.1 | 89.72 | 112,228 |
| LoRA_r4_a8_d0.1 | Yes | 4 | 8 | 0.1 | 89.69 | 112,228 |
| LoRA_r8_a2_d0.1 | Yes | 8 | 2 | 0.1 | 89.49 | 185,956 |
| LoRA_r8_a4_d0.1 | Yes | 8 | 4 | 0.1 | 89.54 | 185,956 |
| LoRA_r8_a8_d0.1 | Yes | 8 | 8 | 0.1 | 89.74 | 185,956 |

Best Q1 test result: **LoRA_r8_a8_d0.1**, Test Accuracy = **89.74%**.

### Optuna Best Configuration

From `outputs_q1/results/optuna_best_config.json`:

- Trial: 12
- Best validation accuracy: 89.38%
- Params:
  - rank: 8
  - alpha: 8
  - lora_dropout: 0.15
  - lr: 0.0002929626

## Q2 Results Summary

### FGSM Comparison (From outputs_q2/results/fgsm_comparison.json)

Clean Accuracy: **93.16%**

| Epsilon | FGSM Scratch Accuracy (%) | FGSM IBM ART Accuracy (%) |
|---:|---:|---:|
| 0.01 | 67.02 | 31.05 |
| 0.03 | 33.39 | 17.89 |
| 0.05 | 22.46 | 14.47 |
| 0.10 | 14.67 | 10.66 |
| 0.20 | 11.43 | 9.99 |
| 0.30 | 10.09 | 10.00 |

### Detector Results (From outputs_q2/results/detector_results.json)

- PGD detector accuracy: **99.925%**
- BIM detector accuracy: **99.475%**

Both satisfy the assignment threshold (>= 70%).

## Generated Artifacts

### Q1

- Checkpoints: `outputs_q1/checkpoints/*.pt`
- Curves and plots: `outputs_q1/plots/*_curves.png`, `*_classwise.png`, `*_gradients.png`
- Results JSON: `outputs_q1/results/*.json`
- HF export folder: `outputs_q1/hf_model/`

### Q2

- Base and detector checkpoints: `outputs_q2/checkpoints/*.pt`
- FGSM and detector plots:
  - `outputs_q2/plots/fgsm_comparison_eps0.01.png`
  - `outputs_q2/plots/perturbation_vs_accuracy.png`
  - `outputs_q2/plots/adversarial_gallery.png`
  - `outputs_q2/plots/cm_pgd.png`
  - `outputs_q2/plots/cm_bim.png`
- Results JSON:
  - `outputs_q2/results/fgsm_comparison.json`
  - `outputs_q2/results/detector_results.json`
