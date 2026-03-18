# v1.1.0  -  Ray Tune Optimised Model Weights ✓ Recommended

This folder holds the **best EN→HI Transformer** checkpoint found by Ray Tune + Optuna,
trained for **50 epochs**  -  achieving **+10.6% BLEU** over the baseline in half the epochs.

> The `.pth` file is **not stored in Git** (216 MB). Download from Hugging Face below.

## Download

| | |
|---|---|
| **HF Model Card** | [priyadip/en-hi-transformer](https://huggingface.co/priyadip/en-hi-transformer) |
| **Direct download** | [m25csa023_ass_4_best_model.pth](https://huggingface.co/priyadip/en-hi-transformer/resolve/main/v1.1.0/m25csa023_ass_4_best_model.pth) |
| **Size** | ~216 MB |

### Auto-download (from repo root)

```bash
python download_model.py
```

### Manual Python download

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id  = "priyadip/en-hi-transformer",
    filename = "v1.1.0/m25csa023_ass_4_best_model.pth",
)
```

## Model Specs

| Property | Value |
|---|---|
| Version | **v1.1.0** |
| Epochs | **50** |
| BLEU Score | **0.8369** (+10.6% vs baseline) |
| Final Loss | 0.1264 |
| Training Time | 13.53 min |
| Learning Rate | **1.112e-4** |
| Batch Size | **32** |
| d_ff | **2560** |
| Dropout | **0.081** |
| Gradient Clipping | max_norm = 1.0 |
| HP Search | Ray Tune + OptunaSearch (20 trials) |
| Scheduler | ASHAScheduler (grace=5, factor=3) |
| Hardware | NVIDIA A100 80 GB |

## Epochs to Beat Baseline

The winning config exceeded baseline BLEU (0.7566) at **epoch 10** during the tuning sweep.
