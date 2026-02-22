#!/usr/bin/env python3
"""
scripts/train.py
────────────────
training pipeline:
  1. Download / cache Goodreads review data
  2. Build train / test splits
  3. Tokenize with DistilBERT tokenizer
  4. Fine-tune DistilBERT using HF Trainer API
  5. Save model + tokenizer + label maps locally

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 5 --batch_size 16 --lr 3e-5
"""

import argparse
import json
import os
import sys

# Disable W&B by default
os.environ["WANDB_DISABLED"] = "true"

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Trainer, TrainingArguments

from src.config import (
    CACHE_MODEL_DIR,
    DATA_DIR,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_LABELS,
    RESULTS_DIR,
    SEED,
    TrainConfig,
)
from src.data import build_label_maps, build_train_test_split, download_all_genres
from src.dataset import GenreDataset
from src.model import get_device, load_model, load_tokenizer
from src.utils import compute_metrics, full_classification_report, save_eval_results


def parse_args():
    p = argparse.ArgumentParser(description="Train DistilBERT on Goodreads genres")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--output_dir", type=str, default=CACHE_MODEL_DIR)
    p.add_argument("--data_cache", type=str, default=os.path.join(DATA_DIR, "genre_reviews.pkl"))
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  TRAINING PIPELINE  —  device: {device}")
    print(f"{'='*60}\n")

    # ── 1. Data ──────────────────────────────────────────────
    print("[Step 1/6] Loading data...")
    genre_reviews = download_all_genres(cache_path=args.data_cache)
    train_texts, train_labels, test_texts, test_labels = build_train_test_split(genre_reviews)

    # ── 2. Label maps ────────────────────────────────────────
    print("[Step 2/6] Building label maps...")
    label2id, id2label = build_label_maps(train_labels)
    print(f"  Labels: {list(label2id.keys())}")

    # ── 3. Tokenize ──────────────────────────────────────────
    print("[Step 3/6] Tokenizing...")
    tokenizer = load_tokenizer(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]

    train_dataset = GenreDataset(train_encodings, train_labels_enc)
    test_dataset = GenreDataset(test_encodings, test_labels_enc)

    # ── 4. Model ─────────────────────────────────────────────
    print("[Step 4/6] Loading model...")
    model = load_model(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        device=device,
    )

    # ── 5. Train ─────────────────────────────────────────────
    print("[Step 5/6] Training...")
    tcfg = TrainConfig()
    training_args = TrainingArguments(
        output_dir=tcfg.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        learning_rate=args.lr,
        warmup_steps=tcfg.warmup_steps,
        weight_decay=tcfg.weight_decay,
        logging_steps=tcfg.logging_steps,
        eval_strategy=tcfg.eval_strategy,
        eval_steps=tcfg.eval_steps,
        save_strategy=tcfg.save_strategy,
        load_best_model_at_end=tcfg.load_best_model_at_end,
        metric_for_best_model=tcfg.metric_for_best_model,
        report_to=tcfg.report_to,
        seed=tcfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── 6. Save ──────────────────────────────────────────────
    print("[Step 6/6] Saving model & artifacts...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save label maps alongside model
    label_maps = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with open(os.path.join(args.output_dir, "label_maps.json"), "w") as f:
        json.dump(label_maps, f, indent=2)

    # Quick eval & persist
    print("\n[Post-training evaluation]")
    eval_results = trainer.evaluate()
    print(eval_results)
    save_eval_results(eval_results, source="local_post_train", output_dir=RESULTS_DIR)

    # Detailed report
    preds = trainer.predict(test_dataset)
    pred_ids = preds.predictions.argmax(-1).flatten().tolist()
    pred_labels = [id2label[i] for i in pred_ids]
    full_classification_report(
        test_labels,
        pred_labels,
        output_path=os.path.join(RESULTS_DIR, "classification_report_local.txt"),
    )

    print(f"\n  Model saved to {args.output_dir}")
    print(f" Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
