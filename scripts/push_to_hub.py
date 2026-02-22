#!/usr/bin/env python3
"""
scripts/push_to_hub.py
──────────────────────
Push the locally saved fine-tuned model, tokenizer, and config to HF Hub.


Usage:
    python scripts/push_to_hub.py --repo_id username/goodreads-genre-classifier
    python scripts/push_to_hub.py --repo_id username/goodreads-genre-classifier --model_dir ./saved_model
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.config import CACHE_MODEL_DIR, HF_REPO_ID


def parse_args():
    p = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    p.add_argument("--model_dir", type=str, default=CACHE_MODEL_DIR)
    p.add_argument("--repo_id", type=str, default=HF_REPO_ID,
                    help="HF Hub repo id, e.g. username/repo-name")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_dir} ...")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)

    print(f"Pushing to Hub → {args.repo_id} ...")
    model.push_to_hub(args.repo_id, commit_message="Upload fine-tuned DistilBERT genre classifier")
    tokenizer.push_to_hub(args.repo_id, commit_message="Upload tokenizer")

    # Also push label_maps.json if it exists
    label_map_path = os.path.join(args.model_dir, "label_maps.json")
    if os.path.exists(label_map_path):
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=label_map_path,
            path_in_repo="label_maps.json",
            repo_id=args.repo_id,
            commit_message="Upload label maps",
        )
        print("  ↳ label_maps.json uploaded")

    print(f"\n  Model is live at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
