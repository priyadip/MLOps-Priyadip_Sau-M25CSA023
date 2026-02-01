"""
Main Training Script for ConvNeXt-V2-Style CNN on CIFAR-10
Includes gradient flow and weight update visualization with WandB logging
"""

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import wandb

from model import ConvNeXtV2Tiny, count_flops_params
from dataloader import create_dataloaders, CIFAR10_MEAN, CIFAR10_STD
from utils import (
    GradientFlowTracker, 
    WeightUpdateTracker,
    plot_training_curves,
    plot_gradient_flow,
    plot_weight_update_flow,
    plot_layerwise_analysis,
    plot_confusion_matrix
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, 
                    gradient_tracker, weight_tracker, epoch, use_amp=True):
    """Train for one epoch with gradient and weight tracking."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_grad_norms = []
    epoch_weight_updates = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        
        # Track gradient norms every 50 batches
        if batch_idx % 50 == 0:
            scaler.unscale_(optimizer)
            grad_norms = gradient_tracker.get_layer_gradient_norms()
            epoch_grad_norms.append(grad_norms)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track weight updates every 50 batches
        if batch_idx % 50 == 0:
            weight_updates = weight_tracker.get_layer_weight_updates()
            epoch_weight_updates.append(weight_updates)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Average gradient norms and weight updates
    avg_grad_norms = {}
    avg_weight_updates = {}
    
    if epoch_grad_norms:
        for key in epoch_grad_norms[0].keys():
            avg_grad_norms[key] = np.mean([g[key] for g in epoch_grad_norms])
    
    if epoch_weight_updates:
        for key in epoch_weight_updates[0].keys():
            avg_weight_updates[key] = np.mean([w[key] for w in epoch_weight_updates])
    
    return train_loss, train_acc, avg_grad_norms, avg_weight_updates


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp=True):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (total // val_loader.batch_size + 1),
            'acc': 100. * correct / total
        })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, 
          scaler, device, config, gradient_tracker, weight_tracker):
    """Full training loop with WandB logging."""
    best_val_acc = 0.0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'gradient_norms': [],
        'weight_updates': [],
    }
    
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"FLOPs: {config['flops']/1e6:.2f} MFLOPs")
    print(f"Parameters: {config['params']:,}")
    print("=" * 60)
    
    for epoch in range(config['epochs']):
        train_loss, train_acc, grad_norms, weight_updates = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            gradient_tracker, weight_tracker, epoch, config['amp']
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, config['amp'])
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['gradient_norms'].append(grad_norms)
        history['weight_updates'].append(weight_updates)
        
        # Log to WandB
        wandb_log = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'learning_rate': current_lr,
        }
        
        for layer_name, norm in grad_norms.items():
            wandb_log[f'gradients/{layer_name}'] = norm
        
        for layer_name, update in weight_updates.items():
            wandb_log[f'weight_updates/{layer_name}'] = update
        
        wandb.log(wandb_log)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            wandb.run.summary['best_val_accuracy'] = best_val_acc
            print(f" New best validation accuracy: {best_val_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 60)
    
    return history, best_model_state, best_val_acc


@torch.no_grad()
def final_evaluation(model, val_loader, device, class_names, save_dir='.', use_amp=True):
    """Final evaluation with confusion matrix and per-class accuracy."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(val_loader, desc='Final Evaluation'):
        inputs = inputs.to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(inputs)
        
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, save_dir)
    
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    class_correct = np.diag(cm)
    class_total = cm.sum(axis=1)
    class_acc = class_correct / class_total * 100
    
    print("\n" + "=" * 60)
    print("Per-Class Accuracy")
    print("=" * 60)
    for name, acc in zip(class_names, class_acc):
        print(f"{name:12}: {acc:.2f}%")
    
    return cm, class_acc


def print_summary(config, history, best_val_acc):
    """Print final summary and findings."""
    print("=" * 70)
    print("                      FINAL SUMMARY AND FINDINGS")
    print("=" * 70)
    
    print(f"\n MODEL SPECIFICATIONS:")
    print(f"   • Architecture: ConvNeXt-V2-Style CNN (CIFAR-10 adapted)")
    print(f"   • Parameters: {config['params']:,}")
    print(f"   • FLOPs: {config['flops']/1e6:.2f} MFLOPs")
    print(f"   • MACs: {config['macs']/1e6:.2f} MMACs")
    
    print(f"\n TRAINING CONFIGURATION:")
    print(f"   • Optimizer: AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
    print(f"   • Scheduler: CosineAnnealingLR")
    print(f"   • Epochs: {config['epochs']}")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"   • Mixed Precision: {'Enabled' if config['amp'] else 'Disabled'}")
    
    print(f"\n RESULTS:")
    print(f"   • Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   • Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   • Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"   • Final Validation Loss: {history['val_loss'][-1]:.4f}")
    
    print("\n" + "=" * 70)


def main(args):
    """Main function."""
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Create dataloaders
    print("\n" + "=" * 60)
    print("Loading CIFAR-10 Dataset")
    print("=" * 60)
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    model = ConvNeXtV2Tiny(
        num_classes=10, 
        dims=[64, 128, 256], 
        depths=[3, 3, 9]
    )
    model = model.to(device)
    
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Count FLOPs
    flops, params, macs = count_flops_params(model, device=device)
    
    # Config
    config = {
        'model_name': 'ConvNeXt-V2-Tiny-CIFAR10',
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'amp': args.amp,
        'dims': [32, 64, 128],
        'depths': [2, 2, 2],
        'flops': flops,
        'macs': macs,
        'params': params,
        'seed': args.seed,
    }
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=f"convnext-tiny-{config['epochs']}epochs",
        config=config,
        tags=["cifar10", "convnext", "low-flops"]
    )
    wandb.watch(model, log='all', log_freq=100)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    scaler = GradScaler(enabled=config['amp'])
    
    # Initialize trackers
    gradient_tracker = GradientFlowTracker(model)
    weight_tracker = WeightUpdateTracker(model)
    
    # Train
    history, best_model_state, best_val_acc = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, device, config, gradient_tracker, weight_tracker
    )
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.models_dir, 'convnext_cifar10_best.pth')
    checkpoint = {
        'model_state_dict': best_model_state,
        'config': config,
        'best_val_acc': best_val_acc,
        'history': history,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nModel checkpoint saved to {checkpoint_path}")
    
    # Log model artifact
    artifact = wandb.Artifact('convnext-cifar10-model', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    plot_training_curves(history, args.results_dir)
    plot_gradient_flow(history, args.results_dir)
    plot_weight_update_flow(history, args.results_dir)
    plot_layerwise_analysis(history, args.results_dir)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    class_names = val_dataset.classes
    cm, class_acc = final_evaluation(
        model, val_loader, device, class_names, 
        args.results_dir, config['amp']
    )
    
    # Print summary
    print_summary(config, history, best_val_acc)
    
    # Log final metrics to WandB
    wandb.run.summary['final_train_acc'] = history['train_acc'][-1]
    wandb.run.summary['final_val_acc'] = history['val_acc'][-1]
    wandb.run.summary['final_train_loss'] = history['train_loss'][-1]
    wandb.run.summary['final_val_loss'] = history['val_loss'][-1]
    
    # Create summary table
    summary_table = wandb.Table(
        columns=['Metric', 'Value'],
        data=[
            ['Model', 'ConvNeXt-V2-Tiny'],
            ['Parameters', f"{config['params']:,}"],
            ['FLOPs', f"{config['flops']/1e6:.2f} MFLOPs"],
            ['Best Val Accuracy', f"{best_val_acc:.2f}%"],
            ['Epochs', f"{config['epochs']}"],
            ['Optimizer', 'AdamW'],
            ['Scheduler', 'CosineAnnealingLR'],
        ]
    )
    wandb.log({'summary_table': summary_table})
    
    wandb.finish()
    print("\n Training complete! All results saved and logged to WandB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ConvNeXt-V2 on CIFAR-10')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory for saving results')
    parser.add_argument('--models-dir', type=str, default='./models',
                        help='Directory for saving model checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=4e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='amp',
                        help='Disable automatic mixed precision')
    
    # WandB arguments
    parser.add_argument('--wandb-project', type=str, default='cifar10-convnext-v2',
                        help='WandB project name')
    
    args = parser.parse_args()
    
    main(args)