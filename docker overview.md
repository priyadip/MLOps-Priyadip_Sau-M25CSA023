# DLOps Assignment 5 Docker Image

This repository provides the Docker image for my DLOps Assignment-5 experiments.

## Author

Name: Priyadip Sau  
Roll Number: M25CSA023

## What This Image Contains

- Q1: ViT-S + LoRA fine-tuning on CIFAR-100
- Q1 Optuna: LoRA hyperparameter search
- Q2: IBM ART adversarial attack and detection experiments on CIFAR-10
- Training, evaluation, plotting, and result generation scripts

## Main Repository and Artifacts

- GitHub (Assignment-5 branch): https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/tree/Assignment-5
- Hugging Face model (best Q1): https://huggingface.co/priyadip/vit-lora-cifar100
- WandB Q1: https://wandb.ai/saupriyadip571-indian-institute-of-technology-jodhpur/DLOps-Ass5-Q1
- WandB Q1 Optuna: https://wandb.ai/saupriyadip571-indian-institute-of-technology-jodhpur/DLOps-Ass5-Q1-Optuna
- WandB Q2: https://wandb.ai/saupriyadip571-indian-institute-of-technology-jodhpur/DLOps-Ass5-Q2

## Docker Pull

```bash
docker pull priyadipsau/dlops-ass5:assignment-5
```

Optional latest tag:

```bash
docker pull priyadipsau/dlops-ass5:latest
```

## Docker Run (GPU)

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  --shm-size=8g \
  priyadipsau/dlops-ass5:assignment-5
```

## Notes

- Designed for GPU-based execution.
- Shared GPU environments may require selecting a less busy GPU.
- For reproducibility, use tag: assignment-5.
