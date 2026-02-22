"""
Custom PyTorch Dataset for tokenized text classification data.
"""

import torch
from torch.utils.data import Dataset


class GenreDataset(Dataset):
    """Wraps HuggingFace tokenizer encodings + integer labels into a PyTorch Dataset."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
