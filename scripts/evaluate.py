#!/usr/bin/env python3
"""
scripts/evaluate.py
───────────────────
Evaluate a fine-tuned model .

Usage:
    # Evaluate local model
    python scripts/evaluate.py --model_path ./saved_model

    # Evaluate model from HF Hub
    python scripts/evaluate.py --model_path your-username/goodreads-genre-classifier --source hub
"""

import argparse
import json
import os
import sys

os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Trainer, TrainingArguments

from src.config import CACHE_MODEL_DIR, DATA_DIR, MAX_LENGTH, RESULTS_DIR
from src.data import build_label_maps, build_train_test_split, download_all_genres
from src.dataset import GenreDataset
from src.model import get_device, load_model, load_tokenizer
from src.utils import compute_metrics, full_classification_report, save_eval_results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned DistilBERT")
    p.add_argument("--model_path", type=str, default=CACHE_MODEL_DIR,
                    help="Local path or HF Hub repo id (e.g. user/repo)")
    p.add_argument("--source", type=str, default="local", choices=["local", "hub"],
                    help="Where to load model from")
    p.add_argument("--data_cache", type=str, default=os.path.join(DATA_DIR, "genre_reviews.pkl"))
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  EVALUATION PIPELINE  —  source: {args.source}")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────
    print("[1/4] Loading data...")
    genre_reviews = download_all_genres(cache_path=args.data_cache)
    train_texts, train_labels, test_texts, test_labels = build_train_test_split(genre_reviews)

    # ── Label maps ───────────────────────────────────────────
    print("[2/4] Building label maps...")
    label2id, id2label = build_label_maps(train_labels)

    # ── Tokenize test data ───────────────────────────────────
    print("[3/4] Tokenizing test data...")
    tokenizer = load_tokenizer(args.model_path)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_labels_enc = [label2id[y] for y in test_labels]
    test_dataset = GenreDataset(test_encodings, test_labels_enc)

    # ── Load model ───────────────────────────────────────────
    print("[4/4] Loading model & evaluating...")
    model = load_model(
        args.model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        device=device,
    )

    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Trainer.evaluate()
    eval_results = trainer.evaluate()
    print("\nTrainer eval results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    result_path = save_eval_results(eval_results, source=args.source, output_dir=RESULTS_DIR)

    # Detailed classification report
    preds = trainer.predict(test_dataset)
    pred_ids = preds.predictions.argmax(-1).flatten().tolist()
    pred_labels = [id2label[i] for i in pred_ids]
    full_classification_report(
        test_labels,
        pred_labels,
        output_path=os.path.join(RESULTS_DIR, f"classification_report_{args.source}.txt"),
    )

    print(f"\n  Evaluation complete — results in {RESULTS_DIR}")
    return result_path


if __name__ == "__main__":
    main()
