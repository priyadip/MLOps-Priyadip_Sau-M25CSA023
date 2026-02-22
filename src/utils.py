"""
Utility functions — metrics, evaluation helpers, result I/O.
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import RESULTS_DIR


# ─────────────────── HuggingFace Trainer metric fn ──────────
def compute_metrics(pred):
    """Compute accuracy + macro-F1 (passed to Trainer)."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


# ─────────────────── Detailed evaluation report ─────────────
def full_classification_report(
    true_labels,
    predicted_labels,
    output_path: str | None = None,
) -> str:
    """Print and optionally save a sklearn classification report."""
    report = classification_report(true_labels, predicted_labels)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved → {output_path}")

    return report


def save_eval_results(
    eval_output: dict,
    source: str = "local",
    output_dir: str = RESULTS_DIR,
) -> str:
    """Persist evaluation metrics as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"eval_{source}_{ts}.json"
    path = os.path.join(output_dir, fname)

    payload = {
        "source": source,
        "timestamp": ts,
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in eval_output.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Eval results saved → {path}")
    return path


def compare_eval_results(path_a: str, path_b: str):
    """Load two eval JSON files and print a side-by-side comparison."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print("\n" + "=" * 60)
    print(f"COMPARISON: {a['source']} vs {b['source']}")
    print("=" * 60)
    all_keys = sorted(set(a["metrics"]) | set(b["metrics"]))
    for k in all_keys:
        va = a["metrics"].get(k, "N/A")
        vb = b["metrics"].get(k, "N/A")
        print(f"  {k:30s}  {str(va):>12s}  {str(vb):>12s}")
