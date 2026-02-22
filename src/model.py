"""
Model and tokenizer loading helpers.
"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.config import MODEL_NAME


def get_device() -> torch.device:
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_name: str = MODEL_NAME) -> DistilBertTokenizerFast:
    """Load the fast tokenizer for DistilBERT."""
    print(f"Loading tokenizer: {model_name}")
    return DistilBertTokenizerFast.from_pretrained(model_name)


def load_model(
    model_name: str = MODEL_NAME,
    num_labels: int = 8,
    id2label: dict | None = None,
    label2id: dict | None = None,
    device: torch.device | None = None,
) -> DistilBertForSequenceClassification:
    """
    Load DistilBERT for sequence classification.
    If *model_name* is a local path or HF repo, it loads the fine-tuned weights.
    """
    device = device or get_device()
    print(f"Loading model: {model_name}  (device={device})")

    kwargs = {"num_labels": num_labels}
    if id2label:
        kwargs["id2label"] = id2label
    if label2id:
        kwargs["label2id"] = label2id

    model = DistilBertForSequenceClassification.from_pretrained(model_name, **kwargs)
    model = model.to(device)
    return model
