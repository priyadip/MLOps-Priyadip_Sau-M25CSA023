"""
ViT-S Finetuning on CIFAR-100 with and without LoRA.

Usage:
  # Baseline (no LoRA):
  python q1_train.py --mode baseline --epochs 10

  # LoRA experiment:
  python q1_train.py --mode lora --rank 4 --alpha 8 --lora_dropout 0.1 --epochs 10

    # Optional: LoRA + partially trainable backbone (last N blocks)
    python q1_train.py --mode lora_partial --rank 8 --alpha 8 --lora_dropout 0.1 \
        --partial_unfreeze_last_n 2 --epochs 10

  # Run all LoRA combinations:
  python q1_train.py --mode all --epochs 10
"""

import argparse
import os
import json
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import timm
from peft import LoraConfig, get_peft_model
import wandb
from tqdm import tqdm

from utils import (
    set_seed, get_device, save_checkpoint,
    plot_training_curves, plot_classwise_histogram,
    plot_gradient_norms, compute_classwise_accuracy,
    count_parameters, CIFAR100_CLASSES,
)


def get_transforms():
    """Get train and test transforms for CIFAR-100 → ViT-S (224x224)."""
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    return train_transform, test_transform


def get_dataloaders(batch_size=64, num_workers=4, val_split=0.1):
    """Get CIFAR-100 dataloaders with train/val/test splits."""
    train_transform, test_transform = get_transforms()

    full_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    # For validation, we use test transform (no augmentation)
    full_train_val = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=test_transform
    )

    num_val = int(len(full_train) * val_split)
    num_train = len(full_train) - num_val

    # Use same split indices for both
    generator = torch.Generator().manual_seed(42)
    train_indices = list(range(num_train))
    val_indices = list(range(num_train, num_train + num_val))

    train_subset = torch.utils.data.Subset(full_train, train_indices)
    val_subset = torch.utils.data.Subset(full_train_val, val_indices)

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_model(
    num_classes=100,
    use_lora=False,
    rank=4,
    alpha=8,
    lora_dropout=0.1,
    partial_unfreeze_last_n=0,
):
    """Create ViT-S model with optional LoRA."""
    # Load pre-trained ViT-S from timm
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)

    if not use_lora:
        # Freeze everything except classification head
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        return model

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Optional setting: unfreeze a small suffix of transformer blocks.
    if partial_unfreeze_last_n > 0:
        n_blocks = len(model.blocks)
        n_unfreeze = min(partial_unfreeze_last_n, n_blocks)
        for block in model.blocks[n_blocks - n_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True
        # Keep final norm trainable with partially trainable blocks.
        for param in model.norm.parameters():
            param.requires_grad = True

    # Unfreeze classification head
    for param in model.head.parameters():
        param.requires_grad = True

    # Apply LoRA to Q, K, V attention weights
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        target_modules=["qkv"],  # timm ViT uses fused qkv projection
        bias="none",
        modules_to_save=["head"],  # Keep classification head trainable
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    gradient_norms=None,
    scaler=None,
    use_amp=False,
):
    """Train for one epoch, optionally tracking gradient norms."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Track gradient norms for LoRA weights
        if gradient_norms is not None:
            for name, param in model.named_parameters():
                if "lora" in name.lower() and param.grad is not None:
                    norm = param.grad.norm().item()
                    if name not in gradient_norms:
                        gradient_norms[name] = []
                    gradient_norms[name].append(norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def run_experiment(
    args,
    use_lora,
    rank=0,
    alpha=0,
    lora_dropout=0.1,
    partial_unfreeze_last_n=0,
    exp_name=None,
):
    """Run a single training experiment."""
    set_seed(42)
    device = get_device()
    print(f"\nDevice: {device}")

    if exp_name is None:
        if use_lora:
            base = f"LoRA_r{rank}_a{alpha}_d{lora_dropout}"
            if partial_unfreeze_last_n > 0:
                exp_name = f"{base}_partial{partial_unfreeze_last_n}"
            else:
                exp_name = base
        else:
            exp_name = "Baseline_NoLoRA"

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=exp_name,
        config={
            "model": "vit_small_patch16_224",
            "dataset": "CIFAR-100",
            "use_lora": use_lora,
            "rank": rank,
            "alpha": alpha,
            "lora_dropout": lora_dropout,
            "partial_unfreeze_last_n": partial_unfreeze_last_n,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "amp": args.amp,
        },
        reinit=True,
    )

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Model
    model = create_model(
        num_classes=100, use_lora=use_lora,
        rank=rank, alpha=alpha, lora_dropout=lora_dropout,
        partial_unfreeze_last_n=partial_unfreeze_last_n,
    )
    model = model.to(device)

    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")
    wandb.config.update({"trainable_params": trainable_params, "total_params": total_params})

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP enabled: {use_amp}")

    # Training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    gradient_norms = {} if use_lora else None
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_norms,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"{exp_name}_best.pt")
            save_checkpoint(model, optimizer, epoch,
                            {"val_acc": val_acc, "val_loss": val_loss}, ckpt_path)

    # Testing
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    # Class-wise accuracy
    class_accs = compute_classwise_accuracy(model, test_loader, 100, device)
    hist_path = os.path.join(args.output_dir, "plots", f"{exp_name}_classwise.png")
    plot_classwise_histogram(class_accs, CIFAR100_CLASSES, hist_path,
                             title=f"Class-wise Test Accuracy - {exp_name}")

    # Log class-wise histogram to WandB
    wandb.log({"classwise_accuracy": wandb.Image(hist_path)})
    class_acc_table = wandb.Table(
        columns=["class", "accuracy"],
        data=[[CIFAR100_CLASSES[i], class_accs[i]] for i in range(100)]
    )
    wandb.log({"classwise_table": class_acc_table})

    # Training curves
    curve_path = os.path.join(args.output_dir, "plots", f"{exp_name}_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curve_path)
    wandb.log({"training_curves": wandb.Image(curve_path)})

    # Gradient norms plot (LoRA only)
    if gradient_norms:
        grad_path = os.path.join(args.output_dir, "plots", f"{exp_name}_gradients.png")
        plot_gradient_norms(gradient_norms, grad_path)
        wandb.log({"gradient_norms": wandb.Image(grad_path)})

    # Save results
    results = {
        "experiment": exp_name,
        "use_lora": use_lora,
        "rank": rank,
        "alpha": alpha,
        "lora_dropout": lora_dropout,
        "partial_unfreeze_last_n": partial_unfreeze_last_n,
        "test_accuracy": test_acc,
        "trainable_params": trainable_params,
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }
    results_path = os.path.join(args.output_dir, "results", f"{exp_name}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    wandb.finish()
    return results


def run_all_experiments(args):
    """Run baseline + all LoRA combinations."""
    all_results = []

    # Baseline: no LoRA
    print("=" * 60)
    print("Running Baseline (No LoRA)")
    print("=" * 60)
    res = run_experiment(args, use_lora=False)
    all_results.append(res)

    # LoRA experiments: ranks x alphas
    ranks = [2, 4, 8]
    alphas = [2, 4, 8]
    dropout = 0.1

    for r, a in itertools.product(ranks, alphas):
        print("=" * 60)
        print(f"Running LoRA: Rank={r}, Alpha={a}, Dropout={dropout}")
        print("=" * 60)
        res = run_experiment(args, use_lora=True, rank=r, alpha=a, lora_dropout=dropout)
        all_results.append(res)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 80)
    print(f"{'Experiment':<30} {'LoRA':<6} {'Rank':<6} {'Alpha':<6} {'Dropout':<8} "
          f"{'Test Acc':<10} {'Trainable Params':<18}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['experiment']:<30} {'Yes' if r['use_lora'] else 'No':<6} "
              f"{r['rank']:<6} {r['alpha']:<6} {r['lora_dropout']:<8} "
              f"{r['test_accuracy']:<10.2f} {r['trainable_params']:<18,}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "results", "all_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Q1: ViT-S + LoRA on CIFAR-100")
    parser.add_argument("--mode", type=str, choices=["baseline", "lora", "lora_partial", "all"],
                        default="all", help="Experiment mode")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (for single lora mode)")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha (for single lora mode)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--partial_unfreeze_last_n",
        type=int,
        default=2,
        help="For mode=lora_partial: number of final ViT blocks to unfreeze",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="outputs_q1", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="DLOps-Ass5-Q1",
                        help="WandB project name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "baseline":
        run_experiment(args, use_lora=False)
    elif args.mode == "lora":
        run_experiment(args, use_lora=True,
                       rank=args.rank, alpha=args.alpha, lora_dropout=args.lora_dropout)
    elif args.mode == "lora_partial":
        run_experiment(
            args,
            use_lora=True,
            rank=args.rank,
            alpha=args.alpha,
            lora_dropout=args.lora_dropout,
            partial_unfreeze_last_n=max(0, args.partial_unfreeze_last_n),
        )
    elif args.mode == "all":
        run_all_experiments(args)


if __name__ == "__main__":
    main()
