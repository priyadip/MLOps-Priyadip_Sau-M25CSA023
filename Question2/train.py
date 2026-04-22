"""
Q2 — Train UNet on the CityScape dataset.

Usage:
    python train.py --data_dir data --epochs 20 --batch_size 16

Outputs:
    checkpoints/unet_best.pt
    plots/train_loss.png, plots/miou.png, plots/mdice.png
    history.json
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CityscapesDataset, collect_paths
from metrics import SegMetrics
from unet_model import UNet


N_CLASSES = 23
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    metric = SegMetrics(n_classes)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        metric.update(logits, masks)
    return metric.compute()


def train_one_epoch(model, loader, optimizer, criterion, device, n_classes):
    model.train()
    losses = []
    metric = SegMetrics(n_classes)
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        metric.update(logits, masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    miou, mdice = metric.compute()
    return float(np.mean(losses)), miou, mdice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # ----- Find dataset root ------------------------------------------------
    # Dataset layout is expected to have 'CameraRGB' and 'CameraMask' somewhere.
    root_candidates = [
        args.data_dir,
        os.path.join(args.data_dir, "data"),
    ]
    for cand in root_candidates:
        if os.path.isdir(os.path.join(cand, "CameraRGB")) and os.path.isdir(
            os.path.join(cand, "CameraMask")
        ):
            data_root = cand
            break
    else:
        # fall back: walk looking for CameraRGB
        data_root = None
        for dp, dn, _ in os.walk(args.data_dir):
            if "CameraRGB" in dn and "CameraMask" in dn:
                data_root = dp
                break
        if data_root is None:
            raise FileNotFoundError("Could not find CameraRGB/CameraMask folders")

    print(f"[info] data_root: {data_root}")

    image_paths, mask_paths = collect_paths(data_root)
    print(f"[info] {len(image_paths)} image/mask pairs")

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=SEED
    )
    print(f"[info] train={len(train_imgs)}  test={len(test_imgs)}")

    train_ds = CityscapesDataset(train_imgs, train_masks)
    test_ds = CityscapesDataset(test_imgs, test_masks)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = UNet(n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)

    history = {"train_loss": [], "train_miou": [], "train_mdice": [],
               "val_miou": [], "val_mdice": []}
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        loss, tr_miou, tr_mdice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, N_CLASSES
        )
        val_miou, val_mdice = evaluate(model, test_loader, device, N_CLASSES)
        scheduler.step()

        history["train_loss"].append(loss)
        history["train_miou"].append(tr_miou)
        history["train_mdice"].append(tr_mdice)
        history["val_miou"].append(val_miou)
        history["val_mdice"].append(val_mdice)

        print(
            f"[epoch {epoch:02d}] loss={loss:.4f} "
            f"train mIoU={tr_miou:.4f} mDice={tr_mdice:.4f} | "
            f"val mIoU={val_miou:.4f} mDice={val_mdice:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_miou": val_miou,
                 "val_mdice": val_mdice},
                os.path.join(args.out_dir, "checkpoints", "unet_best.pt"),
            )
            print(f"[info] saved best checkpoint (mIoU={val_miou:.4f})")

    # ----- Save plots -------------------------------------------------------
    epochs = list(range(1, args.epochs + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss"); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "plots", "train_loss.png")); plt.close()

    plt.figure()
    plt.plot(epochs, history["train_miou"], marker="o", label="train")
    plt.plot(epochs, history["val_miou"], marker="s", label="val")
    plt.title("mIoU"); plt.xlabel("Epoch"); plt.ylabel("mIoU"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "plots", "miou.png")); plt.close()

    plt.figure()
    plt.plot(epochs, history["train_mdice"], marker="o", label="train")
    plt.plot(epochs, history["val_mdice"], marker="s", label="val")
    plt.title("mDice"); plt.xlabel("Epoch"); plt.ylabel("mDice"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "plots", "mdice.png")); plt.close()

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ----- Final test evaluation on best checkpoint --------------------------
    best = torch.load(os.path.join(args.out_dir, "checkpoints", "unet_best.pt"),
                      map_location=device)
    model.load_state_dict(best["model"])
    final_miou, final_mdice = evaluate(model, test_loader, device, N_CLASSES)
    print(f"\n[FINAL TEST] mIoU={final_miou:.4f}  mDice={final_mdice:.4f}")

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump({"mIoU": final_miou, "mDice": final_mdice}, f, indent=2)


if __name__ == "__main__":
    main()
