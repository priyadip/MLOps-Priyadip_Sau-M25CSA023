"""
Q2 — mIoU and mDice for multi-class semantic segmentation using confusion matrix.

The confusion matrix approach is standard and matches what Cityscapes / KITTI
evaluation kits do. Classes that never appear in either prediction or ground
truth are ignored in the mean (otherwise they'd contribute 0 and unfairly tank
the metric).
"""

import numpy as np
import torch


class SegMetrics:
    def __init__(self, n_classes: int):
        self.n = n_classes
        self.cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits : (B, C, H, W)
        targets: (B, H, W) int64 in [0, n_classes-1]
        """
        preds = logits.argmax(dim=1).detach().cpu().numpy().ravel()
        gts = targets.detach().cpu().numpy().ravel()
        mask = (gts >= 0) & (gts < self.n)
        preds, gts = preds[mask], gts[mask]
        idx = gts * self.n + preds
        binc = np.bincount(idx, minlength=self.n * self.n)
        self.cm += binc.reshape(self.n, self.n)

    def reset(self):
        self.cm.fill(0)

    def compute(self):
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Only average over classes that appeared
        present = (tp + fn) > 0

        iou_denom = tp + fp + fn
        dice_denom = 2 * tp + fp + fn

        iou = np.where(iou_denom > 0, tp / np.maximum(iou_denom, 1), 0.0)
        dice = np.where(dice_denom > 0, 2 * tp / np.maximum(dice_denom, 1), 0.0)

        miou = iou[present].mean() if present.any() else 0.0
        mdice = dice[present].mean() if present.any() else 0.0
        return float(miou), float(mdice)
