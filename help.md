# DLOps Assignment 5 — Complete Help Guide

## Table of Contents
1. [Environment Setup (Docker)](#1-environment-setup-docker)
2. [WandB Login](#2-wandb-login)
3. [HuggingFace Login](#3-huggingface-login)
4. [Q1: ViT-S + LoRA on CIFAR-100](#4-q1-vit-s--lora-on-cifar-100)
5. [Q2: Adversarial Attacks (IBM ART)](#5-q2-adversarial-attacks-ibm-art)
6. [GitHub Setup](#6-github-setup)
7. [File Structure](#7-file-structure)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup (Docker)

### Build the Docker image
```bash
docker build -t dlops-ass5 .
```

### Run with GPU support
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  --shm-size=8g \
  dlops-ass5
```

> **Note:** `--shm-size=8g` prevents DataLoader shared memory errors.  
> If you don't have a GPU, remove `--gpus all` (training will be very slow).

### Verify GPU inside container
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 2. WandB Login

### One-time setup
1. Create account at [https://wandb.ai](https://wandb.ai)
2. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Login inside the Docker container:

```bash
wandb login
# Paste your API key when prompted
```

Or set it as an environment variable:
```bash
export WANDB_API_KEY="your-api-key-here"
```

Or pass it when running Docker:
```bash
docker run --gpus all -it --rm \
  -e WANDB_API_KEY="your-api-key-here" \
  -v $(pwd):/workspace \
  --shm-size=8g \
  dlops-ass5
```

### WandB Projects Created
- `DLOps-Ass5-Q1` — All Q1 ViT-S + LoRA experiments
- `DLOps-Ass5-Q1-Optuna` — Optuna hyperparameter search
- `DLOps-Ass5-Q2` — All Q2 adversarial attack experiments

---

## 3. HuggingFace Login

### One-time setup
1. Create account at [https://huggingface.co](https://huggingface.co)
2. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (select **Write** access)
3. Login inside Docker:

```bash
huggingface-cli login
# Paste your token when prompted
```

Or set as environment variable:
```bash
export HF_TOKEN="hf_your-token-here"
```

---

## 4. Q1: ViT-S + LoRA on CIFAR-100

### Step 1 & 2: Run ALL experiments (baseline + 9 LoRA combos)
This runs the baseline (no LoRA) and all 9 LoRA combinations (Rank ∈ {2,4,8} × Alpha ∈ {2,4,8}, Dropout=0.1):

```bash
python q1_train.py --mode all --epochs 10 --batch_size 64 --lr 1e-3
```

**What this produces:**
- Training/validation loss and accuracy logged to WandB per epoch
- Class-wise test accuracy histograms
- Gradient norm plots for LoRA weights
- Best model checkpoints in `outputs_q1/checkpoints/`
- JSON results in `outputs_q1/results/`

### Run individual experiments (optional)

Baseline only:
```bash
python q1_train.py --mode baseline --epochs 10
```

Single LoRA config:
```bash
python q1_train.py --mode lora --rank 4 --alpha 8 --lora_dropout 0.1 --epochs 10
```

### Step 5: Optuna Hyperparameter Search
```bash
python q1_optuna.py --n_trials 20 --epochs 10
```

This searches over rank, alpha, dropout, and learning rate. Best config is saved to `outputs_q1/results/optuna_best_config.json`.

### Step 6: Test Best Model & Push to HuggingFace

After identifying the best model (check WandB or `outputs_q1/results/all_results_summary.json`):

```bash
# Replace with your actual best checkpoint and config
python q1_test.py \
  --checkpoint outputs_q1/checkpoints/LoRA_r4_a8_d0.1_best.pt \
  --use_lora --rank 4 --alpha 8 --lora_dropout 0.1 \
  --push_hf \
  --hf_repo priyadip/vit-lora-cifar100
```

> HuggingFace repo is set to `priyadip/vit-lora-cifar100`.

---

## 5. Q2: Adversarial Attacks (IBM ART)

### Part (i): FGSM Attack Comparison — Full Pipeline
Trains ResNet18 on CIFAR-10 from scratch, then runs FGSM attacks from scratch and via IBM ART:

```bash
python q2_fgsm.py --mode full --epochs 30 --batch_size 128
```

**What this produces:**
- ResNet18 trained to ≥72% on CIFAR-10
- FGSM attacks at multiple epsilon values (0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
- Visual comparison images: original vs FGSM-scratch vs FGSM-ART
- Perturbation strength vs accuracy plot
- 10 sample adversarial images logged to WandB

Train only:
```bash
python q2_fgsm.py --mode train --epochs 30
```

Attack only (requires existing checkpoint):
```bash
python q2_fgsm.py --mode attack --checkpoint outputs_q2/checkpoints/resnet18_cifar10_best.pt
```

### Part (ii): Adversarial Detector — Full Pipeline
Generates PGD and BIM adversarial examples, trains ResNet-34 detectors:

```bash
python q2_detector.py --mode full --epochs_base 30 --epochs_detector 20
```

> **Note:** If you already trained the base ResNet18 in Part (i), you can skip re-training:
```bash
python q2_detector.py --mode detector \
  --checkpoint outputs_q2/checkpoints/resnet18_cifar10_best.pt \
  --epochs_detector 20
```

**What this produces:**
- PGD and BIM adversarial examples via IBM ART
- ResNet-34 binary detectors (clean vs adversarial) for each attack
- Confusion matrices, classification reports
- 10 sample images (clean + FGSM + PGD + BIM) logged to WandB
- Detection accuracy ≥70% for both attacks

---

## 6. GitHub Setup

### Create the branch
```bash
git checkout -b "Assignment 5"
```

### Files to push
```
├── Dockerfile
├── requirements.txt
├── README.md                      # (create from this help + results)
├── utils.py
├── q1_train.py
├── q1_optuna.py
├── q1_test.py
├── q2_fgsm.py
├── q2_detector.py
├── outputs_q1/
│   ├── checkpoints/               # Best model weights for Q1
│   │   └── <best_model>.pt
│   ├── plots/                     # Training curves, histograms, gradients
│   └── results/                   # JSON results
├── outputs_q2/
│   ├── checkpoints/               # All weights: ResNet18, PGD detector, BIM detector
│   │   ├── resnet18_cifar10_best.pt
│   │   ├── detector_pgd_best.pt
│   │   └── detector_bim_best.pt
│   ├── plots/                     # Comparison images, confusion matrices
│   └── results/                   # JSON results
└── report/
    └── Rollnumber_Name_Ass5.pdf
```

### Push commands
```bash
git add .
git commit -m "Assignment 5: ViT-S LoRA + IBM ART Adversarial Attacks"
git push origin "Assignment 5"
```

### README.md content checklist
- [ ] Installation instructions (Docker build + run)
- [ ] Commands to run training and testing for Q1 and Q2
- [ ] Train-val tables and graphs for Q1 and Q2
- [ ] Qualitative image results from Q2 (adversarial samples)
- [ ] WandB project links
- [ ] HuggingFace model link

---

## 7. File Structure

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker container definition |
| `requirements.txt` | Python dependencies |
| `utils.py` | Shared utilities (seed, plotting, metrics) |
| `q1_train.py` | Q1 main: baseline + LoRA experiments on ViT-S/CIFAR-100 |
| `q1_optuna.py` | Q1 Step 5: Optuna hyperparameter optimization |
| `q1_test.py` | Q1 Step 6: Test best model + push to HuggingFace |
| `q2_fgsm.py` | Q2(i): FGSM from scratch vs IBM ART on CIFAR-10 |
| `q2_detector.py` | Q2(ii): PGD/BIM adversarial detectors using ResNet-34 |

---

## 8. Troubleshooting

### CUDA out of memory
Reduce batch size:
```bash
python q1_train.py --mode all --batch_size 32 --epochs 10
python q2_fgsm.py --mode full --batch_size 64 --epochs 30
```

### DataLoader workers crashing
Reduce workers or increase shared memory:
```bash
python q1_train.py --mode all --num_workers 2
# OR increase Docker shm: --shm-size=16g
```

### WandB offline mode
If no internet inside Docker:
```bash
export WANDB_MODE=offline
# Run experiments...
# After exiting, sync:
wandb sync outputs_q1/wandb/run-*
```

### IBM ART import errors
Ensure ART is installed correctly:
```bash
pip install adversarial-robustness-toolbox --break-system-packages
```

### timm ViT-S not found
The model name is `vit_small_patch16_224`. Verify:
```bash
python -c "import timm; print('vit_small_patch16_224' in timm.list_models())"
```

### LoRA target_modules mismatch
The timm ViT uses fused `qkv` projection. If PEFT raises an error, the code uses `target_modules=["qkv"]`. If your timm version uses separate q/k/v, change to:
```python
target_modules=["q_proj", "k_proj", "v_proj"]
# or
target_modules=["attn.qkv"]
```
Check your model's named modules:
```bash
python -c "
import timm
m = timm.create_model('vit_small_patch16_224', pretrained=False)
for n, _ in m.named_modules():
    if 'attn' in n: print(n)
"
```

### LoRA + PEFT modules_to_save
The `modules_to_save=["head"]` in LoraConfig keeps the classification head trainable alongside LoRA. If PEFT complains, remove it and manually unfreeze:
```python
lora_config = LoraConfig(r=rank, lora_alpha=alpha, ...)  # without modules_to_save
model = get_peft_model(model, lora_config)
for param in model.base_model.model.head.parameters():
    param.requires_grad = True
```

---

## Quick-Start Summary (Run Everything)

```bash
# 1. Build and enter Docker
docker build -t dlops-ass5 .
docker run --gpus all -it --rm -v $(pwd):/workspace --shm-size=8g dlops-ass5

# 2. Login
wandb login
huggingface-cli login

# 3. Q1: All experiments + Optuna + Push to HF
python q1_train.py --mode all --epochs 10
python q1_optuna.py --n_trials 20 --epochs 10
python q1_test.py --checkpoint outputs_q1/checkpoints/BEST_MODEL.pt \
  --use_lora --rank R --alpha A --lora_dropout 0.1 \
  --push_hf --hf_repo priyadip/vit-lora-cifar100

# 4. Q2: FGSM + Detectors
python q2_fgsm.py --mode full --epochs 30
python q2_detector.py --mode detector \
  --checkpoint outputs_q2/checkpoints/resnet18_cifar10_best.pt \
  --epochs_detector 20

# 5. Push to GitHub
git checkout -b "Assignment 5"
git add .
git commit -m "Assignment 5 complete"
git push origin "Assignment 5"
```

Replace `BEST_MODEL.pt`, `R`, and `A` with your actual values after running the experiments.
