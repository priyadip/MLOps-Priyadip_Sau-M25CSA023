"""
Q2 Part (i): FGSM Attack - From Scratch vs IBM ART on CIFAR-10.

Usage:
  # Train ResNet18 on CIFAR-10
  python q2_fgsm.py --mode train --epochs 30

  # Run FGSM attacks and comparison
  python q2_fgsm.py --mode attack --checkpoint outputs_q2/checkpoints/resnet18_cifar10_best.pt

  # Full pipeline
  python q2_fgsm.py --mode full --epochs 30
"""

import argparse
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from utils import set_seed, get_device, save_checkpoint, CIFAR10_CLASSES


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """Get CIFAR-10 dataloaders."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                              download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False,
                                             download=True, transform=test_transform)

    # Also get unnormalized test set for ART
    test_set_raw = torchvision.datasets.CIFAR10(root="./data", train=False,
                                                 download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader_raw = DataLoader(test_set_raw, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, test_loader_raw


def create_resnet18(num_classes=10):
    """Create ResNet18 from scratch (not pretrained) for CIFAR-10."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Modify first conv for 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    return model


# ─── Training ───────────────────────────────────────────────────────

def train_resnet18(args):
    """Train ResNet18 on clean CIFAR-10."""
    set_seed(42)
    device = get_device()

    wandb.init(project=args.wandb_project, name="ResNet18_CIFAR10_Train",
               config={"epochs": args.epochs, "lr": 0.1, "batch_size": args.batch_size},
               reinit=True)

    train_loader, test_loader, _ = get_cifar10_loaders(args.batch_size, args.num_workers)

    model = create_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                test_correct += pred.eq(labels).sum().item()
                test_total += labels.size(0)
        test_acc = 100.0 * test_correct / test_total

        scheduler.step()
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Acc={test_acc:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss,
                    "train_acc": train_acc, "test_acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(args.output_dir, "checkpoints", "resnet18_cifar10_best.pt")
            save_checkpoint(model, optimizer, epoch, {"test_acc": test_acc}, ckpt_path)

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    wandb.finish()
    return model


# ─── FGSM From Scratch ─────────────────────────────────────────────

def fgsm_attack_scratch(model, images, labels, epsilon, device):
    """FGSM attack implemented from scratch."""
    images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    # FGSM: perturb in direction of gradient sign
    perturbation = epsilon * images.grad.sign()
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)  # Keep in valid range

    return adv_images.detach()


# ─── FGSM With IBM ART ─────────────────────────────────────────────

def fgsm_attack_art(model, test_loader_raw, epsilon, device):
    """FGSM attack using IBM ART."""
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import PyTorchClassifier

    # Wrap model for ART (expects unnormalized input [0,1])
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.2470, 0.2435, 0.2616]).reshape((3, 1, 1))

    # Create a wrapper model that normalizes internally
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

    fgsm = FastGradientMethod(estimator=classifier, eps=epsilon, batch_size=128)

    # Collect test data (raw, [0,1] range)
    all_images, all_labels = [], []
    for images, labels in test_loader_raw:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Generate adversarial examples
    adv_images = fgsm.generate(x=all_images)

    return all_images, adv_images, all_labels, classifier


# ─── Evaluation and Comparison ──────────────────────────────────────

def evaluate_accuracy(model, images_tensor, labels_tensor, device, normalize=True):
    """Evaluate model accuracy on a batch of images."""
    model.eval()
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)

    correct, total = 0, 0
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(images_tensor), batch_size):
            imgs = images_tensor[i:i + batch_size].to(device)
            lbls = labels_tensor[i:i + batch_size].to(device)
            if normalize:
                imgs = (imgs - mean) / std
            outputs = model(imgs)
            _, pred = outputs.max(1)
            correct += pred.eq(lbls).sum().item()
            total += lbls.size(0)
    return 100.0 * correct / total


def visualize_comparison(clean_imgs, adv_scratch, adv_art, labels, predictions_scratch,
                         predictions_art, save_path, n=10):
    """Visualize original vs adversarial images."""
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 7))

    for i in range(n):
        # Original
        img = clean_imgs[i].transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].set_title(f"{CIFAR10_CLASSES[labels[i]]}", fontsize=7)
        axes[0, i].axis("off")

        # FGSM scratch
        img_s = adv_scratch[i].transpose(1, 2, 0)
        axes[1, i].imshow(np.clip(img_s, 0, 1))
        axes[1, i].set_title(f"{CIFAR10_CLASSES[predictions_scratch[i]]}", fontsize=7,
                             color="red" if predictions_scratch[i] != labels[i] else "green")
        axes[1, i].axis("off")

        # FGSM ART
        img_a = adv_art[i].transpose(1, 2, 0)
        axes[2, i].imshow(np.clip(img_a, 0, 1))
        axes[2, i].set_title(f"{CIFAR10_CLASSES[predictions_art[i]]}", fontsize=7,
                             color="red" if predictions_art[i] != labels[i] else "green")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("FGSM\n(Scratch)", fontsize=10)
    axes[2, 0].set_ylabel("FGSM\n(IBM ART)", fontsize=10)

    plt.suptitle("FGSM Attack Comparison", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visual comparison saved to {save_path}")


def run_attack_comparison(args, model=None):
    """Run FGSM attacks and compare."""
    set_seed(42)
    device = get_device()

    if model is None:
        model = create_resnet18(num_classes=10).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    _, test_loader, test_loader_raw = get_cifar10_loaders(args.batch_size, args.num_workers)

    wandb.init(project=args.wandb_project, name="FGSM_Attack_Comparison",
               config={"epsilons": args.epsilons}, reinit=True)

    # Collect raw test data for scratch attack
    all_raw_images, all_labels_list = [], []
    for images, labels in test_loader_raw:
        all_raw_images.append(images)
        all_labels_list.append(labels)
    all_raw_images = torch.cat(all_raw_images, dim=0)
    all_raw_labels = torch.cat(all_labels_list, dim=0)

    # Clean accuracy
    clean_acc = evaluate_accuracy(model, all_raw_images, all_raw_labels, device, normalize=True)
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")

    results = {"clean_accuracy": clean_acc, "epsilons": {}}

    for eps in args.epsilons:
        print(f"\n--- Epsilon = {eps} ---")

        # FGSM from scratch (process in batches on raw [0,1] images)
        adv_scratch_list = []
        for i in range(0, len(all_raw_images), 256):
            batch_imgs = all_raw_images[i:i + 256]
            batch_lbls = all_raw_labels[i:i + 256]

            # Normalize for model input, then attack
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
            norm_imgs = (batch_imgs - mean) / std

            norm_imgs = norm_imgs.to(device).requires_grad_(True)
            batch_lbls_d = batch_lbls.to(device)

            outputs = model(norm_imgs)
            loss = nn.CrossEntropyLoss()(outputs, batch_lbls_d)
            model.zero_grad()
            loss.backward()

            # Perturb in raw space
            grad_sign = norm_imgs.grad.sign()
            adv_norm = norm_imgs + eps * grad_sign
            # Unnormalize back to [0,1]
            adv_raw = adv_norm * std.to(device) + mean.to(device)
            adv_raw = torch.clamp(adv_raw, 0, 1)
            adv_scratch_list.append(adv_raw.detach().cpu())

        adv_scratch_all = torch.cat(adv_scratch_list, dim=0)
        scratch_acc = evaluate_accuracy(model, adv_scratch_all, all_raw_labels, device,
                                        normalize=True)
        print(f"  FGSM (Scratch) Accuracy: {scratch_acc:.2f}%")

        # FGSM via IBM ART
        clean_np, adv_art_np, labels_np, classifier = fgsm_attack_art(
            model, test_loader_raw, eps, device
        )
        # Evaluate ART adversarial examples
        adv_art_tensor = torch.tensor(adv_art_np, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_np, dtype=torch.long)
        art_acc = evaluate_accuracy(model, adv_art_tensor, labels_tensor, device, normalize=True)
        print(f"  FGSM (IBM ART) Accuracy: {art_acc:.2f}%")

        results["epsilons"][str(eps)] = {
            "scratch_accuracy": scratch_acc,
            "art_accuracy": art_acc,
            "clean_accuracy": clean_acc,
        }

        wandb.log({
            f"eps_{eps}/clean_acc": clean_acc,
            f"eps_{eps}/scratch_acc": scratch_acc,
            f"eps_{eps}/art_acc": art_acc,
        })

        # Visual comparison for first epsilon
        if eps == args.epsilons[0]:
            # Get predictions
            with torch.no_grad():
                mean_t = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
                std_t = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)

                sample_scratch = adv_scratch_all[:10].to(device)
                pred_scratch = model((sample_scratch - mean_t) / std_t).argmax(1).cpu().numpy()

                sample_art = adv_art_tensor[:10].to(device)
                pred_art = model((sample_art - mean_t) / std_t).argmax(1).cpu().numpy()

            vis_path = os.path.join(args.output_dir, "plots", f"fgsm_comparison_eps{eps}.png")
            visualize_comparison(
                clean_np[:10], adv_scratch_all[:10].numpy(), adv_art_np[:10],
                labels_np[:10], pred_scratch, pred_art, vis_path
            )
            wandb.log({"fgsm_visual_comparison": wandb.Image(vis_path)})

            # Log 10 sample pairs to WandB
            wandb_images = []
            for i in range(10):
                clean_img = clean_np[i].transpose(1, 2, 0)
                adv_s = adv_scratch_all[i].numpy().transpose(1, 2, 0)
                adv_a = adv_art_np[i].transpose(1, 2, 0)
                wandb_images.append(wandb.Image(np.clip(clean_img, 0, 1),
                                                caption=f"Clean: {CIFAR10_CLASSES[labels_np[i]]}"))
                wandb_images.append(wandb.Image(np.clip(adv_s, 0, 1),
                                                caption=f"FGSM-Scratch: {CIFAR10_CLASSES[pred_scratch[i]]}"))
                wandb_images.append(wandb.Image(np.clip(adv_a, 0, 1),
                                                caption=f"FGSM-ART: {CIFAR10_CLASSES[pred_art[i]]}"))
            wandb.log({"fgsm_samples": wandb_images})

    # Perturbation vs accuracy plot
    eps_list = [float(e) for e in results["epsilons"].keys()]
    clean_accs = [results["clean_accuracy"]] * len(eps_list)
    scratch_accs = [results["epsilons"][str(e)]["scratch_accuracy"] for e in eps_list]
    art_accs = [results["epsilons"][str(e)]["art_accuracy"] for e in eps_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_list, clean_accs, "g-o", label="Clean", markersize=6)
    ax.plot(eps_list, scratch_accs, "r-s", label="FGSM (Scratch)", markersize=6)
    ax.plot(eps_list, art_accs, "b-^", label="FGSM (IBM ART)", markersize=6)
    ax.set_xlabel("Perturbation Strength (Epsilon)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Perturbation Strength vs Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pert_path = os.path.join(args.output_dir, "plots", "perturbation_vs_accuracy.png")
    os.makedirs(os.path.dirname(pert_path), exist_ok=True)
    plt.savefig(pert_path, dpi=150, bbox_inches="tight")
    plt.close()
    wandb.log({"perturbation_vs_accuracy": wandb.Image(pert_path)})

    # Save results
    res_path = os.path.join(args.output_dir, "results", "fgsm_comparison.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("FGSM COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Epsilon':<10} {'Clean':<12} {'Scratch':<12} {'IBM ART':<12}")
    print("-" * 46)
    for eps in eps_list:
        e = str(eps)
        print(f"{eps:<10} {results['clean_accuracy']:<12.2f} "
              f"{results['epsilons'][e]['scratch_accuracy']:<12.2f} "
              f"{results['epsilons'][e]['art_accuracy']:<12.2f}")

    wandb.finish()
    return results


def main():
    parser = argparse.ArgumentParser(description="Q2 Part (i): FGSM Attack Comparison")
    parser.add_argument("--mode", type=str, choices=["train", "attack", "full"], default="full")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs_q2/checkpoints/resnet18_cifar10_best.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--output_dir", type=str, default="outputs_q2")
    parser.add_argument("--wandb_project", type=str, default="DLOps-Ass5-Q2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        train_resnet18(args)
    elif args.mode == "attack":
        run_attack_comparison(args)
    elif args.mode == "full":
        model = train_resnet18(args)
        args.checkpoint = os.path.join(args.output_dir, "checkpoints", "resnet18_cifar10_best.pt")
        run_attack_comparison(args, model)


if __name__ == "__main__":
    main()
