"""
Utilities for Gradient Flow and Weight Update Flow Tracking
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb


class GradientFlowTracker:
    """
    Track gradient flow through the network.
    Computes L2 norm of gradients for each layer after backward pass.
    """
    
    def __init__(self, model):
        self.model = model
        self.gradient_norms = {}
    
    def compute_gradient_norms(self):
        """
        Compute L2 norm of gradients for each named parameter.
        Call this after loss.backward().
        """
        gradient_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                gradient_norms[name] = grad_norm
        
        return gradient_norms
    
    def get_layer_gradient_norms(self):
        """
        Get gradient norms aggregated by layer/block.
        """
        grad_norms = self.compute_gradient_norms()
        
        layer_norms = {
            'stem': 0.0,
            'stage_0': 0.0,
            'stage_1': 0.0,
            'stage_2': 0.0,
            'head': 0.0,
            'downsample': 0.0,
        }
        
        counts = {k: 0 for k in layer_norms}
        
        for name, norm in grad_norms.items():
            if 'stem' in name:
                layer_norms['stem'] += norm
                counts['stem'] += 1
            elif 'stages.0' in name:
                layer_norms['stage_0'] += norm
                counts['stage_0'] += 1
            elif 'stages.1' in name:
                layer_norms['stage_1'] += norm
                counts['stage_1'] += 1
            elif 'stages.2' in name:
                layer_norms['stage_2'] += norm
                counts['stage_2'] += 1
            elif 'head' in name:
                layer_norms['head'] += norm
                counts['head'] += 1
            elif 'downsample' in name:
                layer_norms['downsample'] += norm
                counts['downsample'] += 1
        
        for k in layer_norms:
            if counts[k] > 0:
                layer_norms[k] /= counts[k]
        
        return layer_norms


class WeightUpdateTracker:
    """
    Track weight updates ||W_t - W_{t-1}|| across training.
    """
    
    def __init__(self, model):
        self.model = model
        self.previous_weights = {}
        self._store_weights()
    
    def _store_weights(self):
        """Store current weights for comparison."""
        for name, param in self.model.named_parameters():
            self.previous_weights[name] = param.data.clone()
    
    def compute_weight_updates(self):
        """
        Compute L2 norm of weight updates since last call.
        """
        weight_updates = {}
        
        for name, param in self.model.named_parameters():
            if name in self.previous_weights:
                update = param.data - self.previous_weights[name]
                update_norm = update.norm(2).item()
                weight_updates[name] = update_norm
        
        self._store_weights()
        
        return weight_updates
    
    def get_layer_weight_updates(self):
        """
        Get weight updates aggregated by layer/block.
        """
        weight_updates = self.compute_weight_updates()
        
        layer_updates = {
            'stem': 0.0,
            'stage_0': 0.0,
            'stage_1': 0.0,
            'stage_2': 0.0,
            'head': 0.0,
            'downsample': 0.0,
        }
        
        counts = {k: 0 for k in layer_updates}
        
        for name, update in weight_updates.items():
            if 'stem' in name:
                layer_updates['stem'] += update
                counts['stem'] += 1
            elif 'stages.0' in name:
                layer_updates['stage_0'] += update
                counts['stage_0'] += 1
            elif 'stages.1' in name:
                layer_updates['stage_1'] += update
                counts['stage_1'] += 1
            elif 'stages.2' in name:
                layer_updates['stage_2'] += update
                counts['stage_2'] += 1
            elif 'head' in name:
                layer_updates['head'] += update
                counts['head'] += 1
            elif 'downsample' in name:
                layer_updates['downsample'] += update
                counts['downsample'] += 1
        
        for k in layer_updates:
            if counts[k] > 0:
                layer_updates[k] /= counts[k]
        
        return layer_updates


def plot_training_curves(history, save_dir='.'):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90% Target')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule (Cosine Annealing)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy progress
    axes[1, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    axes[1, 1].axhline(y=90, color='g', linestyle='--', linewidth=2, label='90% Target')
    axes[1, 1].fill_between(epochs, 0, history['val_acc'], alpha=0.3, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy Progress')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    save_path = f'{save_dir}/training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    wandb.log({"training_curves": wandb.Image(save_path)})
    print(f"Saved training curves to {save_path}")


def plot_gradient_flow(history, save_dir='.'):
    """
    Visualize gradient flow across layers over training.
    """
    gradient_norms = history['gradient_norms']
    epochs = range(1, len(gradient_norms) + 1)
    layer_names = list(gradient_norms[0].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    
    for i, layer in enumerate(layer_names):
        norms = [g[layer] for g in gradient_norms]
        axes[0].plot(epochs, norms, '-o', color=colors[i], label=layer, 
                     linewidth=2, markersize=4)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Gradient L2 Norm', fontsize=12)
    axes[0].set_title('Gradient Flow: L2 Norm Over Training', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Heatmap
    grad_matrix = np.array([[g[layer] for layer in layer_names] for g in gradient_norms])
    
    im = axes[1].imshow(grad_matrix.T, aspect='auto', cmap='viridis',
                        extent=[1, len(epochs), -0.5, len(layer_names)-0.5])
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Layer', fontsize=12)
    axes[1].set_title('Gradient Flow Heatmap', fontsize=14)
    axes[1].set_yticks(range(len(layer_names)))
    axes[1].set_yticklabels(layer_names)
    plt.colorbar(im, ax=axes[1], label='Gradient L2 Norm')
    
    plt.tight_layout()
    save_path = f'{save_dir}/gradient_flow.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    wandb.log({"gradient_flow": wandb.Image(save_path)})
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Gradient Flow Analysis")
    print("=" * 60)
    for layer in layer_names:
        norms = [g[layer] for g in gradient_norms]
        print(f"{layer:15} | Mean: {np.mean(norms):.6f} | "
              f"Std: {np.std(norms):.6f} | Min: {np.min(norms):.6f} | Max: {np.max(norms):.6f}")


def plot_weight_update_flow(history, save_dir='.'):
    """
    Visualize weight update flow across layers over training.
    """
    weight_updates = history['weight_updates']
    epochs = range(1, len(weight_updates) + 1)
    layer_names = list(weight_updates[0].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line plot
    colors = plt.cm.plasma(np.linspace(0, 1, len(layer_names)))
    
    for i, layer in enumerate(layer_names):
        updates = [w[layer] for w in weight_updates]
        axes[0].plot(epochs, updates, '-o', color=colors[i], label=layer,
                     linewidth=2, markersize=4)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Weight Update L2 Norm', fontsize=12)
    axes[0].set_title('Weight Update Flow: ||W_t - W_{t-1}|| Over Training', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Heatmap
    update_matrix = np.array([[w[layer] for layer in layer_names] for w in weight_updates])
    
    im = axes[1].imshow(update_matrix.T, aspect='auto', cmap='plasma',
                        extent=[1, len(epochs), -0.5, len(layer_names)-0.5])
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Layer', fontsize=12)
    axes[1].set_title('Weight Update Heatmap', fontsize=14)
    axes[1].set_yticks(range(len(layer_names)))
    axes[1].set_yticklabels(layer_names)
    plt.colorbar(im, ax=axes[1], label='Weight Update L2 Norm')
    
    plt.tight_layout()
    save_path = f'{save_dir}/weight_update_flow.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    wandb.log({"weight_update_flow": wandb.Image(save_path)})
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Weight Update Flow Analysis")
    print("=" * 60)
    for layer in layer_names:
        updates = [w[layer] for w in weight_updates]
        print(f"{layer:15} | Mean: {np.mean(updates):.6f} | "
              f"Std: {np.std(updates):.6f} | Min: {np.min(updates):.6f} | Max: {np.max(updates):.6f}")


def plot_layerwise_analysis(history, save_dir='.'):
    """
    Create detailed layer-wise analysis plots.
    """
    gradient_norms = history['gradient_norms']
    weight_updates = history['weight_updates']
    
    layer_names = list(gradient_norms[0].keys())
    epochs = range(1, len(gradient_norms) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, layer in enumerate(layer_names):
        ax = axes[i]
        
        grad_vals = [g[layer] for g in gradient_norms]
        update_vals = [w[layer] for w in weight_updates]
        
        ax.plot(epochs, grad_vals, 'b-o', label='Gradient Norm', linewidth=2, markersize=4)
        ax.plot(epochs, update_vals, 'r-s', label='Weight Update', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2 Norm')
        ax.set_title(f'{layer.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    save_path = f'{save_dir}/layerwise_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    wandb.log({"layerwise_analysis": wandb.Image(save_path)})


def plot_confusion_matrix(cm, class_names, save_dir='.'):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    save_path = f'{save_dir}/confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    wandb.log({"confusion_matrix": wandb.Image(save_path)})
    print(f"Saved confusion matrix to {save_path}")