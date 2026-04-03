"""
Q2 Part (ii): Adversarial Detection Model using ResNet-34 on CIFAR-10.

(a) Detect PGD adversarial images (binary: clean vs adversarial)
(b) Detect BIM adversarial images (binary: clean vs adversarial)

Usage:
  # Full pipeline (train base model, generate adversarial data, train detectors)
  python q2_detector.py --mode full --epochs_base 30 --epochs_detector 20

  # Just train detectors (assumes base model checkpoint exists)
  python q2_detector.py --mode detector --checkpoint outputs_q2/checkpoints/resnet18_cifar10_best.pt
"""

import argparse
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from q2_fgsm import create_resnet18, train_resnet18, get_cifar10_loaders
from utils import set_seed, get_device, save_checkpoint, CIFAR10_CLASSES


def generate_adversarial_data(model, attack_type, test_loader_raw, device, eps=0.03,
                              eps_step=0.01, max_iter=10):
    """Generate adversarial examples using IBM ART (PGD or BIM)."""
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.2470, 0.2435, 0.2616]).reshape((3, 1, 1))

    class NormWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.base_model(x)

    wrapped = NormWrapper(model).to(device).eval()

    classifier = PyTorchClassifier(
        model=wrapped,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    # Choose attack
    if attack_type == "pgd":
        attack = ProjectedGradientDescent(
            estimator=classifier, eps=eps, eps_step=eps_step,
            max_iter=max_iter, batch_size=128
        )
    elif attack_type == "bim":
        attack = BasicIterativeMethod(
            estimator=classifier, eps=eps, eps_step=eps_step,
            max_iter=max_iter, batch_size=128
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # Collect test data
    all_images, all_labels = [], []
    for images, labels in test_loader_raw:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Generating {attack_type.upper()} adversarial examples (eps={eps})...")
    adv_images = attack.generate(x=all_images)

    return all_images, adv_images, all_labels


def create_detector_dataset(clean_images, adv_images):
    """Create binary classification dataset: 0=clean, 1=adversarial."""
    n = len(clean_images)
    # Combine
    all_images = np.concatenate([clean_images, adv_images], axis=0)
    all_labels = np.concatenate([np.zeros(n), np.ones(n)], axis=0)

    # Shuffle
    indices = np.random.permutation(len(all_labels))
    all_images = all_images[indices]
    all_labels = all_labels[indices]

    # Split 80/20
    split = int(0.8 * len(all_labels))
    train_imgs = torch.tensor(all_images[:split], dtype=torch.float32)
    train_lbls = torch.tensor(all_labels[:split], dtype=torch.long)
    test_imgs = torch.tensor(all_images[split:], dtype=torch.float32)
    test_lbls = torch.tensor(all_labels[split:], dtype=torch.long)

    train_dataset = TensorDataset(train_imgs, train_lbls)
    test_dataset = TensorDataset(test_imgs, test_lbls)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    return train_loader, test_loader


def create_resnet34_detector():
    """Create ResNet-34 binary detector (clean vs adversarial)."""
    model = models.resnet34(weights=None, num_classes=2)
    # Modify for 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train_detector(train_loader, test_loader, attack_name, args, device):
    """Train adversarial detector."""
    wandb.init(
        project=args.wandb_project,
        name=f"Detector_{attack_name.upper()}",
        config={"attack": attack_name, "epochs": args.epochs_detector},
        reinit=True,
    )

    model = create_resnet34_detector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_detector)

    best_acc = 0.0
    for epoch in range(1, args.epochs_detector + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"[{attack_name}] Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # Test
        model.eval()
        test_correct, test_total = 0, 0
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                test_correct += pred.eq(labels).sum().item()
                test_total += labels.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        test_acc = 100.0 * test_correct / test_total
        scheduler.step()

        print(f"  [{attack_name}] Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss,
                    "train_acc": train_acc, "test_acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                     f"detector_{attack_name}_best.pt")
            save_checkpoint(model, optimizer, epoch, {"test_acc": test_acc}, ckpt_path)

    print(f"\n  [{attack_name}] Best Detection Accuracy: {best_acc:.2f}%")

    # Classification report
    report = classification_report(all_labels_list, all_preds,
                                   target_names=["Clean", "Adversarial"], output_dict=True)
    print(f"\n{classification_report(all_labels_list, all_preds, target_names=['Clean', 'Adversarial'])}")

    # Confusion matrix
    cm = confusion_matrix(all_labels_list, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Clean", "Adversarial"],
                yticklabels=["Clean", "Adversarial"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {attack_name.upper()} Detector")
    cm_path = os.path.join(args.output_dir, "plots", f"cm_{attack_name}.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    wandb.log({f"confusion_matrix_{attack_name}": wandb.Image(cm_path)})

    wandb.finish()

    return best_acc, report


def log_adversarial_samples(clean_images, adv_images_dict, labels, args):
    """Log 10 samples of clean and adversarial images for each attack to WandB."""
    wandb.init(project=args.wandb_project, name="Adversarial_Samples_Gallery",
               reinit=True)

    wandb_images = []
    for i in range(10):
        clean_img = clean_images[i].transpose(1, 2, 0)
        wandb_images.append(
            wandb.Image(np.clip(clean_img, 0, 1),
                        caption=f"Clean: {CIFAR10_CLASSES[labels[i]]}")
        )
        for attack_name, adv_imgs in adv_images_dict.items():
            adv_img = adv_imgs[i].transpose(1, 2, 0)
            wandb_images.append(
                wandb.Image(np.clip(adv_img, 0, 1),
                            caption=f"{attack_name}: {CIFAR10_CLASSES[labels[i]]}")
            )

    wandb.log({"adversarial_samples": wandb_images})

    # Also save a figure locally
    attacks = list(adv_images_dict.keys())
    n_rows = 1 + len(attacks)
    fig, axes = plt.subplots(n_rows, 10, figsize=(20, 2 * n_rows))

    for i in range(10):
        img = clean_images[i].transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].set_title(CIFAR10_CLASSES[labels[i]], fontsize=7)
        axes[0, i].axis("off")

        for row, attack_name in enumerate(attacks, 1):
            adv_img = adv_images_dict[attack_name][i].transpose(1, 2, 0)
            axes[row, i].imshow(np.clip(adv_img, 0, 1))
            axes[row, i].axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=10)
    for row, name in enumerate(attacks, 1):
        axes[row, 0].set_ylabel(name, fontsize=10)

    plt.suptitle("Clean vs Adversarial Samples", fontsize=14)
    plt.tight_layout()
    gallery_path = os.path.join(args.output_dir, "plots", "adversarial_gallery.png")
    os.makedirs(os.path.dirname(gallery_path), exist_ok=True)
    plt.savefig(gallery_path, dpi=150, bbox_inches="tight")
    plt.close()
    wandb.log({"adversarial_gallery": wandb.Image(gallery_path)})
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Q2 Part (ii): Adversarial Detector")
    parser.add_argument("--mode", type=str, choices=["full", "detector"], default="full")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs_q2/checkpoints/resnet18_cifar10_best.pt")
    parser.add_argument("--epochs_base", type=int, default=30)
    parser.add_argument("--epochs_detector", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs_q2")
    parser.add_argument("--wandb_project", type=str, default="DLOps-Ass5-Q2")
    parser.add_argument("--eps", type=float, default=0.03, help="Attack epsilon")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Get base model
    if args.mode == "full":
        # Reuse training from q2_fgsm or train fresh
        args_train = argparse.Namespace(
            epochs=args.epochs_base, batch_size=args.batch_size,
            num_workers=args.num_workers, output_dir=args.output_dir,
            wandb_project=args.wandb_project
        )
        base_model = train_resnet18(args_train)
        args.checkpoint = os.path.join(args.output_dir, "checkpoints", "resnet18_cifar10_best.pt")
    else:
        base_model = create_resnet18(num_classes=10).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])

    base_model.eval()

    # Step 2: Get raw test loader
    _, _, test_loader_raw = get_cifar10_loaders(args.batch_size, args.num_workers)

    # Step 3: Generate adversarial examples
    # Also generate FGSM for the gallery (requirement)
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.2470, 0.2435, 0.2616]).reshape((3, 1, 1))

    print("\n--- Generating PGD adversarial examples ---")
    clean_pgd, adv_pgd, labels_pgd = generate_adversarial_data(
        base_model, "pgd", test_loader_raw, device, eps=args.eps
    )

    print("\n--- Generating BIM adversarial examples ---")
    clean_bim, adv_bim, labels_bim = generate_adversarial_data(
        base_model, "bim", test_loader_raw, device, eps=args.eps
    )

    # Also generate FGSM for the gallery
    print("\n--- Generating FGSM adversarial examples (for gallery) ---")
    from q2_fgsm import fgsm_attack_art
    clean_fgsm, adv_fgsm, labels_fgsm, _ = fgsm_attack_art(
        base_model, test_loader_raw, args.eps, device
    )

    # Step 4: Log 10 samples to WandB
    adv_dict = {
        "FGSM (ART)": adv_fgsm,
        "PGD": adv_pgd,
        "BIM": adv_bim,
    }
    log_adversarial_samples(clean_pgd, adv_dict, labels_pgd, args)

    # Step 5: Train detectors
    results = {}

    # (a) PGD Detector
    print("\n" + "=" * 60)
    print("Training PGD Adversarial Detector (ResNet-34)")
    print("=" * 60)
    pgd_train_loader, pgd_test_loader = create_detector_dataset(clean_pgd, adv_pgd)
    pgd_acc, pgd_report = train_detector(pgd_train_loader, pgd_test_loader, "pgd", args, device)
    results["pgd"] = {"detection_accuracy": pgd_acc, "report": pgd_report}

    # (b) BIM Detector
    print("\n" + "=" * 60)
    print("Training BIM Adversarial Detector (ResNet-34)")
    print("=" * 60)
    bim_train_loader, bim_test_loader = create_detector_dataset(clean_bim, adv_bim)
    bim_acc, bim_report = train_detector(bim_train_loader, bim_test_loader, "bim", args, device)
    results["bim"] = {"detection_accuracy": bim_acc, "report": bim_report}

    # Summary
    print("\n" + "=" * 60)
    print("ADVERSARIAL DETECTION SUMMARY")
    print("=" * 60)
    print(f"PGD Detection Accuracy:  {pgd_acc:.2f}%")
    print(f"BIM Detection Accuracy:  {bim_acc:.2f}%")

    # Save results
    res_path = os.path.join(args.output_dir, "results", "detector_results.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {res_path}")


if __name__ == "__main__":
    main()
