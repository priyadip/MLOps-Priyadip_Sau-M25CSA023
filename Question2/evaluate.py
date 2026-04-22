"""
Q2 — Evaluate the best checkpoint on the held-out 20% test split.
Prints mIoU and mDice. Reuses the same seed-42 split as train.py.
"""

import argparse
import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import CityscapesDataset, collect_paths
from metrics import SegMetrics
from unet_model import UNet


N_CLASSES = 23
SEED = 42


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    m = SegMetrics(N_CLASSES)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        m.update(model(imgs), masks)
    return m.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt", default="checkpoints/unet_best.pt")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find data root
    data_root = args.data_dir
    for dp, dn, _ in os.walk(args.data_dir):
        if "CameraRGB" in dn and "CameraMask" in dn:
            data_root = dp
            break

    image_paths, mask_paths = collect_paths(data_root)
    _, test_imgs, _, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=SEED
    )

    test_loader = DataLoader(
        CityscapesDataset(test_imgs, test_masks),
        batch_size=args.batch_size, shuffle=False,
    )

    model = UNet(n_classes=N_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    miou, mdice = run(model, test_loader, device)
    print(f"Test mIoU : {miou:.4f}")
    print(f"Test mDice: {mdice:.4f}")

    with open("test_metrics.json", "w") as f:
        json.dump({"mIoU": miou, "mDice": mdice}, f, indent=2)


if __name__ == "__main__":
    main()
