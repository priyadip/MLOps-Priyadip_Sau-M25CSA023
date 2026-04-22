"""
Q2 — Custom Cityscape dataloader.

Masks in CameraMask are RGB where each R channel value (0..22) denotes a class.
The snippet in the exam takes np.max across the channel axis which effectively
collapses an already-single-channel class-index mask. We preserve that behaviour
exactly so the answer matches the snippet.
"""

import os
import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_H, IMG_W = 96, 128


class CityscapesDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str]):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        assert len(image_paths) == len(mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)  # collapse 3 channels → class index

        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        mask = torch.from_numpy(mask).long()          # HW
        return img, mask


def collect_paths(data_root: str) -> Tuple[List[str], List[str]]:
    """Find RGB/CameraMask pairs and keep only matching filenames."""
    rgb_dir = os.path.join(data_root, "CameraRGB")
    mask_dir = os.path.join(data_root, "CameraMask")

    imgs = sorted(glob.glob(os.path.join(rgb_dir, "*")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*")))

    img_map = {os.path.basename(p): p for p in imgs}
    mask_map = {os.path.basename(p): p for p in masks}
    common = sorted(set(img_map) & set(mask_map))
    if not common:
        raise RuntimeError(f"No matched RGB/Mask pairs in {data_root}")

    image_paths = [img_map[k] for k in common]
    mask_paths = [mask_map[k] for k in common]
    return image_paths, mask_paths
