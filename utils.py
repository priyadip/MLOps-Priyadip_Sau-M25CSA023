"""Shared utilities for DLOps Assignment 5."""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"Checkpoint saved to {path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, val_accs, "r-o", label="Val Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_classwise_histogram(class_accs, class_names, save_path, title="Class-wise Test Accuracy"):
    """Plot class-wise accuracy histogram."""
    fig, ax = plt.subplots(figsize=(20, 6))
    x = np.arange(len(class_names))
    bars = ax.bar(x, class_accs, color=sns.color_palette("viridis", len(class_names)))
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Class-wise histogram saved to {save_path}")


def plot_gradient_norms(gradient_norms_dict, save_path):
    """Plot gradient norms for LoRA weights over training steps."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, norms in gradient_norms_dict.items():
        short_name = name.split(".")[-1] if len(name) > 40 else name
        ax.plot(norms, label=short_name, alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms on LoRA Weights During Training")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gradient norms plot saved to {save_path}")


def compute_classwise_accuracy(model, dataloader, num_classes, device):
    """Compute per-class accuracy."""
    model.eval()
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for c in range(num_classes):
                mask = labels == c
                class_correct[c] += (predicted[mask] == labels[mask]).sum().item()
                class_total[c] += mask.sum().item()

    class_accs = []
    for c in range(num_classes):
        if class_total[c] > 0:
            class_accs.append(100.0 * class_correct[c] / class_total[c])
        else:
            class_accs.append(0.0)
    return class_accs


def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
