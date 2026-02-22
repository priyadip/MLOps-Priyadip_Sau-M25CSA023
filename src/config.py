"""
Centralized configuration for the Goodreads Genre Classification .
All hyperparameters, paths, and constants .
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────── Paths ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")

# ─────────────────────────── Model ───────────────────────────
MODEL_NAME = "distilbert-base-cased"

# ─────────────────────────── Data ────────────────────────────
GENRE_URLS: Dict[str, str] = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

NUM_LABELS = len(GENRE_URLS)

# Sampling
HEAD_REVIEWS = 10_000        # read first N reviews per genre from stream
SAMPLE_SIZE = 2_000          # randomly sample from the HEAD_REVIEWS
TRAIN_PER_GENRE = 800        # per-genre train split
TEST_PER_GENRE = 200         # per-genre test split  (sample_size=1000 → 800+200)

MAX_LENGTH = 512             # tokenizer max sequence length
SEED = 42

# ─────────────────── Training Hyper-params ───────────────────
# @dataclass
# class TrainConfig:
#     num_train_epochs: int = 3
#     per_device_train_batch_size: int = 10
#     per_device_eval_batch_size: int = 16
#     learning_rate: float = 5e-5
#     warmup_steps: int = 100
#     weight_decay: float = 0.01
#     logging_steps: int = 10
#     eval_strategy: str = "steps"
#     eval_steps: int = 50
#     save_strategy: str = "steps"
#     load_best_model_at_end: bool = True
#     metric_for_best_model: str = "accuracy"
#     output_dir: str = "./training_output"
#     report_to: str = "none"          # disable wandb by default
#     seed: int = SEED

@dataclass
class TrainConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 10
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    
    # Evaluation & Saving (Keep these aligned!)
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50  # Added to ensure checkpoints exist for best model loading
    
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    output_dir: str = "./training_output"
    report_to: str = "none"
    seed: int = SEED


HF_USERNAME = os.getenv("HF_USERNAME", "priyadip")
HF_REPO_NAME = os.getenv("HF_REPO_NAME", "goodreads-genre-classifier")
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"
