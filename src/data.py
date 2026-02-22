"""
Data loading, sampling, and train/test splitting for Goodreads reviews.
"""

import gzip
import json
import os
import pickle
import random
from typing import Dict, List, Tuple

import requests

from src.config import (
    DATA_DIR,
    GENRE_URLS,
    HEAD_REVIEWS,
    SAMPLE_SIZE,
    SEED,
    TRAIN_PER_GENRE,
)


def load_reviews_from_url(
    url: str,
    head: int = HEAD_REVIEWS,
    sample_size: int = SAMPLE_SIZE,
) -> List[str]:
    """Stream reviews from a gzipped JSON URL and return a random sample."""
    reviews: List[str] = []
    count = 0

    print(f"  ↳ Downloading from {url[:80]}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("review_text", "").strip()
            if text:
                reviews.append(text)
            count += 1
            if head is not None and count >= head:
                break

    if len(reviews) > sample_size:
        random.seed(SEED)
        reviews = random.sample(reviews, sample_size)

    print(f"    Collected {len(reviews)} reviews")
    return reviews


def download_all_genres(
    genre_urls: Dict[str, str] = GENRE_URLS,
    cache_path: str | None = None,
) -> Dict[str, List[str]]:
    """
    Download all genres. If *cache_path* exists, load from cache instead.
    Returns {genre_name: [review_text, ...]}.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    genre_reviews: Dict[str, List[str]] = {}
    for genre, url in genre_urls.items():
        print(f"[{genre}]")
        genre_reviews[genre] = load_reviews_from_url(url)

    # Cache to disk
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(genre_reviews, f)
        print(f"Cached data → {cache_path}")

    return genre_reviews


def build_train_test_split(
    genre_reviews: Dict[str, List[str]],
    train_per_genre: int = TRAIN_PER_GENRE,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split per-genre reviews into train / test lists.
    Returns (train_texts, train_labels, test_texts, test_labels).
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        random.seed(SEED)
        sampled = random.sample(reviews, min(1000, len(reviews)))

        for review in sampled[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_per_genre:]:
            test_texts.append(review)
            test_labels.append(genre)

    print(
        f"Split → Train: {len(train_texts)} | Test: {len(test_texts)}"
    )
    return train_texts, train_labels, test_texts, test_labels


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return (label2id, id2label) dictionaries."""
    unique = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
