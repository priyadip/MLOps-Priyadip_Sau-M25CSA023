#!/usr/bin/env python3
"""
scripts/eval_from_hub.py
────────────────────────
Task 8: Load model from  HF repo and re-evaluate.
Compares metrics with the local evaluation.

Usage:
    python scripts/eval_from_hub.py --repo_id username/goodreads-genre-classifier
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HF_REPO_ID, RESULTS_DIR
from src.utils import compare_eval_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", type=str, default=HF_REPO_ID)
    return p.parse_args()


def main():
    args = parse_args()

    # Reuse evaluate.py with --source hub
    from scripts.evaluate import main as eval_main

    # Monkey-patch sys.argv for the evaluate script
    sys.argv = [
        "evaluate.py",
        "--model_path", args.repo_id,
        "--source", "hub",
    ]
    hub_result_path = eval_main()

    # Find latest local eval to compare
    local_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_local*.json")))
    if local_files:
        print("\n" + "─" * 60)
        compare_eval_results(local_files[-1], hub_result_path)
    else:
        print("\n  No local eval results found for comparison. Run train.py first.")


if __name__ == "__main__":
    main()
