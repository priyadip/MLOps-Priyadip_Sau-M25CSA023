# v1.0.0  -  Baseline Model Weights

This folder holds the **baseline EN→HI Transformer** checkpoint trained for **100 epochs** with fixed hyperparameters.

> The `.pth` file is **not stored in Git** (192 MB). Download from Hugging Face below.

## Download

| | |
|---|---|
| **HF Model Card** | [priyadip/en-hi-transformer](https://huggingface.co/priyadip/en-hi-transformer) |
| **Direct download** | [transformer_translation_final.pth](https://huggingface.co/priyadip/en-hi-transformer/resolve/main/v1.0.0/transformer_translation_final.pth) |
| **Size** | ~192 MB |

### Auto-download (from repo root)

```bash
python download_model.py
```

### Manual Python download

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id  = "priyadip/en-hi-transformer",
    filename = "v1.0.0/transformer_translation_final.pth",
    local_dir= "./transformer_translation_final",
)
```

## Model Specs

| Property | Value |
|---|---|
| Version | **v1.0.0** |
| Epochs | 100 |
| BLEU Score | **0.7566** |
| Final Loss | 0.0998 |
| Training Time | 12.27 min |
| Learning Rate | 1e-4 |
| Batch Size | 60 |
| d_ff | 2048 |
| Dropout | 0.10 |
| Gradient Clipping |  -  |
| Hardware | NVIDIA A100 80 GB |
