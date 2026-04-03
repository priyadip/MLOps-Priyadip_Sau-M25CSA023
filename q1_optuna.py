"""
Q1 Step 5: Optuna hyperparameter search for best LoRA configuration.

Usage:
  python q1_optuna.py --n_trials 20 --epochs 10
"""

import argparse
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
import wandb

from q1_train import get_dataloaders, create_model, train_one_epoch, evaluate
from utils import set_seed, get_device, count_parameters


def objective(trial, args):
    """Optuna objective: maximize validation accuracy."""
    set_seed(42)
    device = get_device()

    # Hyperparameters to search
    rank = trial.suggest_categorical("rank", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    exp_name = f"optuna_trial{trial.number}_r{rank}_a{alpha}"

    wandb.init(
        project=args.wandb_project,
        name=exp_name,
        config={
            "rank": rank, "alpha": alpha,
            "lora_dropout": lora_dropout, "lr": lr,
            "trial_number": trial.number,
        },
        reinit=True,
    )

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = create_model(num_classes=100, use_lora=True,
                         rank=rank, alpha=alpha, lora_dropout=lora_dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        })

        best_val_acc = max(best_val_acc, val_acc)

        # Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.log({"best_val_acc": best_val_acc})
    wandb.finish()
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Q1 Optuna: Find best LoRA config")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs_q1")
    parser.add_argument("--wandb_project", type=str, default="DLOps-Ass5-Q1-Optuna")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        study_name="LoRA_Hyperopt",
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    # Results
    print("\n" + "=" * 60)
    print("OPTUNA STUDY COMPLETE")
    print("=" * 60)

    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Trials: {len(study.trials)} total, {len(complete)} complete, {len(pruned)} pruned")

    best = study.best_trial
    print(f"\nBest Trial #{best.number}:")
    print(f"  Val Accuracy: {best.value:.2f}%")
    print(f"  Params: {best.params}")

    # Save best config
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    best_config = {
        "trial_number": best.number,
        "best_val_accuracy": best.value,
        "params": best.params,
    }
    config_path = os.path.join(args.output_dir, "results", "optuna_best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to {config_path}")

    # Print suggestion
    print(f"\nTo train with best config:")
    print(f"  python q1_train.py --mode lora "
          f"--rank {best.params['rank']} "
          f"--alpha {best.params['alpha']} "
          f"--lora_dropout {best.params['lora_dropout']:.2f} "
          f"--lr {best.params['lr']:.6f} --epochs {args.epochs}")


if __name__ == "__main__":
    main()
