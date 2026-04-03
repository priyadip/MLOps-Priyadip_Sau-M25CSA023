"""
Test best model and push to HuggingFace.

Usage:
    python q1_test.py --checkpoint outputs_q1/checkpoints/BEST_MODEL.pt --push_hf --hf_repo priyadip/vit-lora-cifar100
"""

import argparse
import os
import json
import re
import shutil

import torch
import torch.nn as nn
from huggingface_hub import HfApi, create_repo

from q1_train import get_dataloaders, create_model, evaluate
from utils import (
    set_seed, get_device, compute_classwise_accuracy,
    plot_classwise_histogram, count_parameters, CIFAR100_CLASSES,
)


def infer_config_from_checkpoint(checkpoint_path):
    """Infer model config from checkpoint filename when possible."""
    name = os.path.basename(checkpoint_path)

    # Example: LoRA_r8_a8_d0.1_best.pt
    match = re.match(r"LoRA_r(\d+)_a(\d+)_d([0-9]*\.?[0-9]+)_best\.pt$", name)
    if match:
        return {
            "use_lora": True,
            "rank": int(match.group(1)),
            "alpha": int(match.group(2)),
            "lora_dropout": float(match.group(3)),
        }

    # Example: Baseline_NoLoRA_best.pt
    if name == "Baseline_NoLoRA_best.pt":
        return {
            "use_lora": False,
            "rank": 0,
            "alpha": 0,
            "lora_dropout": 0.1,
        }

    return None


def test_model(args):
    set_seed(42)
    device = get_device()

    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    # Recreate model architecture
    model = create_model(
        num_classes=100,
        use_lora=args.use_lora,
        rank=args.rank,
        alpha=args.alpha,
        lora_dropout=args.lora_dropout,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # Class-wise accuracy
    class_accs = compute_classwise_accuracy(model, test_loader, 100, device)
    hist_path = os.path.join(args.output_dir, "plots", "best_model_classwise.png")
    plot_classwise_histogram(class_accs, CIFAR100_CLASSES, hist_path,
                             title=f"Best Model Class-wise Test Accuracy ({test_acc:.2f}%)")

    trainable = count_parameters(model, trainable_only=True)
    print(f"Trainable params: {trainable:,}")

    return model, test_acc


def push_to_huggingface(args, model):
    """Push model weights to HuggingFace."""
    print(f"\nPushing model to HuggingFace: {args.hf_repo}")

    # Save model weights locally
    save_dir = os.path.join(args.output_dir, "hf_model")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save config
    config = {
        "model": "vit_small_patch16_224",
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "use_lora": args.use_lora,
        "rank": args.rank,
        "alpha": args.alpha,
        "lora_dropout": args.lora_dropout,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy best-run plots into hf_model so the model card can display images on HF.
    exp_name = "Baseline_NoLoRA" if not args.use_lora else f"LoRA_r{args.rank}_a{args.alpha}_d{args.lora_dropout}"
    plot_dir = os.path.join(args.output_dir, "plots")
    candidate_plots = [
        f"{exp_name}_curves.png",
        f"{exp_name}_classwise.png",
        "best_model_classwise.png",
    ]
    for plot_name in candidate_plots:
        src = os.path.join(plot_dir, plot_name)
        dst = os.path.join(save_dir, plot_name)
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Could not copy {plot_name}: {e}")

    # Create repo and upload
    api = HfApi()
    try:
        create_repo(args.hf_repo, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    api.upload_folder(
        folder_path=save_dir,
        repo_id=args.hf_repo,
        repo_type="model",
    )
    print(f"Model pushed to https://huggingface.co/{args.hf_repo}")


def main():
    parser = argparse.ArgumentParser(description="Q1: Test and push to HuggingFace")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--use_lora", action="store_true", help="Model uses LoRA")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs_q1")
    parser.add_argument("--push_hf", action="store_true", help="Push to HuggingFace")
    parser.add_argument("--hf_repo", type=str, default="priyadip/vit-lora-cifar100")
    args = parser.parse_args()

    inferred = infer_config_from_checkpoint(args.checkpoint)
    if inferred is not None:
        args.use_lora = inferred["use_lora"]
        args.rank = inferred["rank"]
        args.alpha = inferred["alpha"]
        args.lora_dropout = inferred["lora_dropout"]
        print(
            "Inferred config from checkpoint: "
            f"use_lora={args.use_lora}, rank={args.rank}, "
            f"alpha={args.alpha}, lora_dropout={args.lora_dropout}"
        )
    else:
        print("Could not infer config from checkpoint filename; using CLI arguments.")

    model, test_acc = test_model(args)

    if args.push_hf:
        push_to_huggingface(args, model)


if __name__ == "__main__":
    main()
