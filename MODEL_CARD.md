---
language: en
tags:
  - image-classification
  - vision-transformer
  - lora
  - peft
  - cifar100
  - pytorch
library_name: pytorch
pipeline_tag: image-classification
license: mit
model-index:
  - name: vit-lora-cifar100
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          type: cifar100
          name: CIFAR-100
        metrics:
          - type: accuracy
            name: Top-1 Accuracy
            value: 89.74
---

# ViT-S + LoRA for CIFAR-100

This model is a Vision Transformer Small (`vit_small_patch16_224`) adapted with LoRA for 100-class image classification on CIFAR-100.

It is intended for users who want to download a ready-to-use classifier and run inference.

## Model At A Glance

- Architecture: ViT-Small (`vit_small_patch16_224`)
- Adaptation: LoRA (PEFT)
- Task: image classification
- Labels: 100 classes
- Dataset: CIFAR-100
- Best configuration: `r=8`, `alpha=8`, `dropout=0.1`

## Model Structure After Fine-Tuning

The uploaded weights are for a ViT-S model with LoRA adapters and a trainable classification head.

1. Backbone: `vit_small_patch16_224` (pretrained ViT-S)
2. LoRA adapters: injected into fused attention `qkv` projection layers
3. Classification head: output dimension updated to 100 classes
4. Parameter-efficient setup: base backbone mostly frozen, LoRA + head trained

### Fine-tuned architecture summary

| Component | Status After Fine-tuning |
|---|---|
| Patch embedding + transformer backbone | Frozen (except LoRA pathways) |
| Attention `qkv` projection | LoRA adapters active |
| MLP blocks | Frozen |
| Classification head | Trainable |
| Output classes | 100 (CIFAR-100) |

The uploaded `model.pt` is a state dict that contains both base model parameters and LoRA parameters.

## Performance

| Metric | Value |
|---|---:|
| Top-1 Accuracy | 89.74% |
| Best Validation Accuracy | 89.14% |
| Trainable Parameters | 185,956 |

## Recommended Use Cases

- Direct image classification inference with a ViT+LoRA model
- Educational demos for Hugging Face model download and inference
- Baseline testing on CIFAR-100-style natural images

## Out-of-Scope Use Cases

- Safety-critical decision making
- Medical/legal/financial decisions
- Deployment to domains very different from CIFAR-style natural images without additional validation

## Limitations

- Evaluated only on CIFAR-100
- Accuracy may drop on real-world or shifted data distributions
- Not designed as a robustness- or fairness-optimized model

## Bias, Risks, and Responsible Use

- Class imbalance and dataset artifacts may influence predictions
- Misclassifications are expected for ambiguous or out-of-distribution inputs
- Always include human review before real-world decisions

## How To Use 

This model repo stores:

- `model.pt` (PyTorch state dict)
- `config.json` (model + LoRA settings)

To use the model, you must recreate the same architecture and then load `model.pt`.

### Install dependencies 

```bash
pip install torch timm peft huggingface_hub torchvision pillow
```

### Download from Hugging Face and load 

```python
import torch
import timm
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download


def build_model(num_classes=100, rank=8, alpha=8, lora_dropout=0.1):
  # 1) Base ViT-S
  model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)

  # 2) Freeze base params
  for p in model.parameters():
    p.requires_grad = False

  # 3) Keep head trainable
  for p in model.head.parameters():
    p.requires_grad = True

  # 4) Attach LoRA to fused qkv projection
  lora_cfg = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    lora_dropout=lora_dropout,
    target_modules=["qkv"],
    bias="none",
    modules_to_save=["head"],
  )
  model = get_peft_model(model, lora_cfg)
  return model

repo_id = "priyadip/vit-lora-cifar100"
weights_path = hf_hub_download(repo_id=repo_id, filename="model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(
  num_classes=100,
  rank=8,
  alpha=8,
  lora_dropout=0.1,
).to(device)

state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
```

### Minimal inference example

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Use the same normalization expected by this model
transform = transforms.Compose([
  transforms.Resize(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
             std=[0.2675, 0.2565, 0.2761]),
])

img = Image.open("your_image.png").convert("RGB")
x = transform(img).unsqueeze(0).to(next(model.parameters()).device)

with torch.no_grad():
  logits = model(x)
  pred = logits.argmax(dim=1).item()

print("Predicted class index:", pred)
```

Note: the output is a CIFAR-100 class index (0-99). Map indices to class names according to your CIFAR-100 label list.

If needed, fetch config directly from the repo:

```python
from huggingface_hub import hf_hub_download
import json

cfg_path = hf_hub_download(repo_id="priyadip/vit-lora-cifar100", filename="config.json")
cfg = json.load(open(cfg_path))
print(cfg)
```

## Visuals

![Training Curves](./LoRA_r8_a8_d0.1_curves.png)

![Class-wise Accuracy](./LoRA_r8_a8_d0.1_classwise.png)

![LoRA Gradient Norms](./LoRA_r8_a8_d0.1_gradients.png)

### What We Can Observe From These Visuals

- **Training curves:** The train/validation curves improve smoothly and stay close, which suggests stable optimization and good generalization for this setup.
- **Class-wise accuracy:** Performance is strong across most CIFAR-100 classes, indicating that adaptation is not concentrated on only a few easy categories.
- **LoRA gradient norms:** Gradients remain active and controlled during training, showing that LoRA adapters receive useful updates without unstable spikes.

### Why LoRA ?

LoRA adds small trainable low-rank adapters inside attention layers instead of updating the full backbone. This gives three practical advantages:

1. **Parameter efficiency:** far fewer trainable parameters than full fine-tuning.
2. **Stable transfer:** pretrained ViT knowledge is largely preserved while adapters specialize to CIFAR-100.
3. **Better compute/memory trade-off:** strong accuracy with lighter optimization cost.

These plots support that LoRA provides an effective balance between accuracy and training efficiency for this model.

## Framework Versions

- PyTorch
- timm
- PEFT (LoRA)

## Citation

- ViT paper (original architecture): https://arxiv.org/abs/2010.11929
- timm library (implementation used): https://github.com/huggingface/pytorch-image-models
- timm ViT-S model entry (reference checkpoint page): https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k

```bibtex
@inproceedings{dosovitskiy2021vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}

@misc{mangrulkar2022peft,
  title={PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author={Mangrulkar, Sourab and Paul, Sayak and others},
  year={2022},
  howpublished={\url{https://github.com/huggingface/peft}}
}

@misc{rw2019timm,
  title={PyTorch Image Models},
  author={Wightman, Ross},
  year={2019},
  howpublished={\url{https://github.com/huggingface/pytorch-image-models}}
}
```
