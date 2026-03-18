#!/usr/bin/env python3
"""
Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna
=========================================================================
Roll No : m25csa023
File    : m25csa023_ass_4_tuned_en_to_hi.py

This script:
 Trains the baseline model (100 epochs, hardcoded hyperparams)
           and records Time, Final Loss, BLEU.
Refactors the training loop for Ray Tune, defines a 5-param
           search space, and runs OptunaSearch + ASHA scheduler.
Retrains the best config for the full epoch budget, evaluates
           BLEU, and saves the best model weights.


Dataset : English-Hindi.tsv  (place in the same directory or set --data_path)
Usage   : python m25csa023_ass_4_tuned_en_to_hi.py [--data_path PATH] [--skip_baseline]
"""

import os
import sys
import time
import math
import json
import pickle
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for GPU servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import ray
import ray.air
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import ray.train

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)


# 0.  GLOBAL CONFIG & ARGUMENT PARSING
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer EN->HI Tuning")
    parser.add_argument("--data_path", type=str, default="English-Hindi.tsv",
                        help="Path to English-Hindi.tsv")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip Part 1 baseline training (if already done)")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of Ray Tune trials")
    parser.add_argument("--tune_epochs", type=int, default=30,
                        help="Max epochs per trial during tuning")
    parser.add_argument("--final_epochs", type=int, default=50,
                        help="Epochs for final retraining of best config")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save outputs")
    return parser.parse_args()


# PLOT STYLING — consistent academic/publication quality
COLORS = {
    "baseline": "#2196F3",    # Blue
    "tuned":    "#FF5722",    # Deep Orange
    "accent1":  "#4CAF50",    # Green
    "accent2":  "#9C27B0",    # Purple
    "accent3":  "#FF9800",    # Orange
    "grid":     "#E0E0E0",
    "bg":       "#FAFAFA",
    "text":     "#212121",
}

def setup_plot_style():
    """Set global matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    COLORS["bg"],
        "axes.edgecolor":    "#BDBDBD",
        "axes.grid":         True,
        "grid.color":        COLORS["grid"],
        "grid.linewidth":    0.6,
        "grid.alpha":        0.7,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    12,
        "legend.fontsize":   10,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#BDBDBD",
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.15,
    })

setup_plot_style()


# 1.  DATA LOADING & VOCABULARY  (identical to en_to_hi.ipynb)
class Vocabulary:
    """Word-level vocabulary with frequency thresholding."""
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    @staticmethod
    def tokenize(sentence):
        return sentence.lower().strip().split()

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])


def encode_sentence(sentence, vocab, max_len=50):
    tokens = ([vocab.stoi["<sos>"]]
              + vocab.numericalize(sentence)[:max_len - 2]
              + [vocab.stoi["<eos>"]])
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))


class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=50):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output


def load_data(data_path):
    """Load & clean English-Hindi TSV, build vocabs, return everything."""
    df = pd.read_csv(data_path, sep="\t", header=None,
                     names=["id1", "en", "id2", "hi"])
    df = df[["en", "hi"]].dropna().reset_index(drop=True)
    print(f"[DATA] Loaded {len(df)} sentence pairs from {data_path}")

    en_vocab = Vocabulary(freq_threshold=2)
    hi_vocab = Vocabulary(freq_threshold=2)
    en_vocab.build_vocab(df["en"].tolist())
    hi_vocab.build_vocab(df["hi"].tolist())
    print(f"[DATA] English vocab: {len(en_vocab)}  |  Hindi vocab: {len(hi_vocab)}")

    return df, en_vocab, hi_vocab


# 2.  TRANSFORMER ARCHITECTURE  (verbatim from en_to_hi.ipynb)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.query_linear(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_linear(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_linear(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(target_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_layers=6, num_heads=8, d_ff=2048, max_len=100,
                 dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers,
                               num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers,
                               num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones((size, size))).bool().to(
            next(self.parameters()).device
        )

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_mask = self.make_subsequent_mask(tgt.size(1))
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)


# 3.  EVALUATION HELPERS
val_dataset = [
    ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?", "आप कैसे हैं?"),
    ("You should sleep.", "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.","मुझे टॉम को बताने दीजिए।")
]

SMOOTHIE = SmoothingFunction().method4


def translate_sentence(model, sentence, en_vocab, hi_vocab, device,
                       max_len=50):
    model.eval()
    src_pad_idx = en_vocab["<pad>"]
    tgt_pad_idx = hi_vocab["<pad>"]
    tokens = encode_sentence(sentence, en_vocab, max_len=max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_tokens = [hi_vocab["<sos>"]]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_pad_idx, tgt_pad_idx)
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break

    translated = [hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]]
    return " ".join(translated)


def evaluate_bleu(model, en_vocab, hi_vocab, device, val_data=None):
    """Compute corpus BLEU on the validation pairs."""
    if val_data is None:
        val_data = val_dataset
    references, hypotheses = [], []
    for en_sent, hi_sent in val_data:
        pred = translate_sentence(model, en_sent, en_vocab, hi_vocab, device)
        hypotheses.append(pred.split())
        references.append([hi_sent.split()])
    score = corpus_bleu(references, hypotheses, smoothing_function=SMOOTHIE)
    return score


# 4.  PART 1 — BASELINE TRAINING  (mirrors en_to_hi.ipynb exactly)
def run_baseline(df, en_vocab, hi_vocab, output_dir, num_epochs=100):
    """
    Train with the original hardcoded hyperparameters for 100 epochs.
    Returns dict with metrics + per-epoch loss/bleu history for plots.
    """
    print("\n" + "=" * 70)
    print("  PART 1 -- BASELINE TRAINING (100 epochs, original hyperparams)")
    print("=" * 70)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {DEVICE}")
    torch.backends.cudnn.benchmark = True

    # --- Original hyperparams (from notebook) ---
    BATCH_SIZE = 60
    MAX_LEN = 50
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    LR = 1e-4

    SRC_PAD_IDX = en_vocab["<pad>"]
    TGT_PAD_IDX = hi_vocab["<pad>"]

    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(en_vocab), tgt_vocab_size=len(hi_vocab),
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        d_ff=D_FF, max_len=MAX_LEN, dropout=DROPOUT
    ).to(DEVICE)
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---- Track history for plots ----
    loss_history = []
    bleu_history = []       # list of (epoch, bleu)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = (src.to(DEVICE), tgt_in.to(DEVICE),
                                    tgt_out.to(DEVICE))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(src, tgt_in, SRC_PAD_IDX, TGT_PAD_IDX)
                loss = criterion(output.view(-1, output.shape[-1]),
                                 tgt_out.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        # Evaluate BLEU every 10 epochs (+ first and last)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            bleu = evaluate_bleu(model, en_vocab, hi_vocab, DEVICE)
            bleu_history.append((epoch + 1, bleu))
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  --  "
                  f"Loss: {avg_loss:.4f}  BLEU: {bleu:.4f}")
        elif (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  --  Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    final_bleu = evaluate_bleu(model, en_vocab, hi_vocab, DEVICE)

    # Save baseline weights
    baseline_path = os.path.join(output_dir, "transformer_translation_final.pth")
    torch.save(model.state_dict(), baseline_path)

    # Save vocabs
    with open(os.path.join(output_dir, "en_vocab.pkl"), "wb") as f:
        pickle.dump(en_vocab, f)
    with open(os.path.join(output_dir, "hi_vocab.pkl"), "wb") as f:
        pickle.dump(hi_vocab, f)

    metrics = {
        "time_sec": round(elapsed, 2),
        "time_min": round(elapsed / 60, 2),
        "final_loss": round(avg_loss, 4),
        "bleu_score": round(final_bleu, 4),
        "epochs": num_epochs,
        "loss_history": [float(x) for x in loss_history],
        "bleu_history": [(int(e), float(b)) for e, b in bleu_history],
    }

    print(f"\n  -- Baseline Results --")
    print(f"  Time       : {metrics['time_min']:.1f} min ({metrics['time_sec']:.0f} s)")
    print(f"  Final Loss : {metrics['final_loss']:.4f}")
    print(f"  BLEU Score : {metrics['bleu_score']:.4f}  ({metrics['bleu_score']*100:.2f}%)")
    print(f"  Saved      : {baseline_path}\n")

    with open(os.path.join(output_dir, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Print sample translations
    print("  -- Sample Baseline Translations --")
    for en, hi in val_dataset[:3]:
        pred = translate_sentence(model, en, en_vocab, hi_vocab, DEVICE)
        print(f"    EN : {en}")
        print(f"    REF: {hi}")
        print(f"    HYP: {pred}")
        print()

    return metrics


# 5.  PART 2 — RAY TUNE + OPTUNA + ASHA
def train_tune(config):
    """
    Training function compatible with Ray Tune.
    Receives hyperparameters via `config` dict.
    Reports loss and BLEU every epoch via tune.report().
    """
    data_path = config["data_path"]
    src_vocab_size = config["src_vocab_size"]
    tgt_vocab_size = config["tgt_vocab_size"]
    max_len = config["max_len"]
    src_pad_idx = config["src_pad_idx"]
    tgt_pad_idx = config["tgt_pad_idx"]
    num_epochs = config["num_epochs"]

    lr = config["lr"]
    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    dropout = config["dropout"]

    d_model = 512
    num_layers = 6

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(data_path, sep="\t", header=None,
                     names=["id1", "en", "id2", "hi"])
    df = df[["en", "hi"]].dropna().reset_index(drop=True)

    en_vocab = Vocabulary(freq_threshold=2)
    hi_vocab = Vocabulary(freq_threshold=2)
    en_vocab.build_vocab(df["en"].tolist())
    hi_vocab.build_vocab(df["hi"].tolist())

    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=4,
                        pin_memory=True)

    model = Transformer(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        d_ff=d_ff, max_len=max_len, dropout=dropout
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for src, tgt_in, tgt_out in loader:
            src = src.to(DEVICE)
            tgt_in = tgt_in.to(DEVICE)
            tgt_out = tgt_out.to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(src, tgt_in, src_pad_idx, tgt_pad_idx)
                loss = criterion(output.view(-1, output.shape[-1]),
                                 tgt_out.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        bleu = 0.0
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            bleu = evaluate_bleu(model, en_vocab, hi_vocab, DEVICE)

        tune.report({"loss": avg_loss, "bleu": bleu, "epoch": epoch + 1})


def run_tuning(df, en_vocab, hi_vocab, args):
    """
    Part 2: Configure search space, run Optuna + ASHA.
    Returns best config AND per-trial data for plots.
    """
    print("\n" + "=" * 70)
    print("  PART 2 -- RAY TUNE + OPTUNA + ASHA HYPERPARAMETER SWEEP")
    print("=" * 70)

    search_space = {
        "lr":         tune.loguniform(5e-5, 5e-4),
        "batch_size": tune.qrandint(32, 40, 2),
        "num_heads":  tune.choice([4, 8, 16]),
        "d_ff":       tune.qrandint(1536, 3072, 512),
        "dropout":    tune.uniform(0.05, 0.25),
        "data_path":       os.path.abspath(args.data_path),
        "src_vocab_size":  len(en_vocab),
        "tgt_vocab_size":  len(hi_vocab),
        "max_len":         50,
        "src_pad_idx":     en_vocab["<pad>"],
        "tgt_pad_idx":     hi_vocab["<pad>"],
        "num_epochs":      args.tune_epochs,
    }

    print(f"  Tunable hyperparameters:")
    print(f"    lr         : loguniform(5e-5, 5e-4)")
    print(f"    batch_size : qrandint(32, 40, 2)")
    print(f"    num_heads  : choice([4, 8, 16])")
    print(f"    d_ff       : qrandint(1536, 3072, 512)")
    print(f"    dropout    : uniform(0.05, 0.25)")
    print(f"  Trials       : {args.num_samples}")
    print(f"  Max epochs/trial : {args.tune_epochs}")
    print()

    optuna_search = OptunaSearch(metric="loss", mode="min")

    asha_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.tune_epochs,
        grace_period=5,
        reduction_factor=3,
    )

    tuner = tune.Tuner(
        tune.with_resources(train_tune,
                            {"gpu": 1 if torch.cuda.is_available() else 0}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=asha_scheduler,
            num_samples=args.num_samples,
        ),
        run_config=ray.air.RunConfig(
            storage_path="/scratch/data/m25csa023/ass4/ray_results",
        ),
        param_space=search_space,
    )

    print("  Starting Ray Tune sweep ...\n")
    results = tuner.fit()

    best_result = results.get_best_result(metric="loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics

    print("\n  -- Best Trial Results --")
    print(f"  Loss   : {best_metrics['loss']:.4f}")
    print(f"  BLEU   : {best_metrics.get('bleu', 'N/A')}")
    print(f"  Epochs : {best_metrics.get('epoch', '?')}")
    print(f"  Config :")
    for k in ["lr", "batch_size", "num_heads", "d_ff", "dropout"]:
        print(f"    {k:12s} = {best_config[k]}")

    saveable_config = {k: best_config[k] for k in
                       ["lr", "batch_size", "num_heads", "d_ff", "dropout"]}
    with open(os.path.join(args.output_dir, "best_config.json"), "w") as f:
        json.dump(saveable_config, f, indent=2)

    # ---- Extract per-trial data for plots ----
    trial_data = []
    for result in results:
        try:
            trial_df = result.metrics_dataframe
            cfg = result.config
            trial_data.append({
                "loss_curve": trial_df["loss"].tolist() if trial_df is not None else [],
                "bleu_curve": trial_df["bleu"].tolist() if trial_df is not None and "bleu" in trial_df else [],
                "final_loss": float(result.metrics.get("loss", float("inf"))),
                "final_bleu": float(result.metrics.get("bleu", 0)),
                "lr": float(cfg.get("lr", 0)),
                "batch_size": int(cfg.get("batch_size", 0)),
                "num_heads": int(cfg.get("num_heads", 0)),
                "d_ff": int(cfg.get("d_ff", 0)),
                "dropout": float(cfg.get("dropout", 0)),
            })
        except Exception:
            pass

    with open(os.path.join(args.output_dir, "trial_data.json"), "w") as f:
        json.dump(trial_data, f, indent=2)

    return saveable_config, trial_data


# 6.  PART 3 — RETRAIN BEST CONFIG & SAVE FINAL MODEL
def retrain_best(df, en_vocab, hi_vocab, best_config, args):
    """
    Retrain with the best config. Returns metrics + history for plots.
    """
    print("\n" + "=" * 70)
    print("  PART 3 -- RETRAINING WITH BEST CONFIG")
    print("=" * 70)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    MAX_LEN = 50

    lr = best_config["lr"]
    batch_size = best_config["batch_size"]
    num_heads = best_config["num_heads"]
    d_ff = best_config["d_ff"]
    dropout = best_config["dropout"]
    d_model = 512
    num_layers = 6
    num_epochs = args.final_epochs

    SRC_PAD_IDX = en_vocab["<pad>"]
    TGT_PAD_IDX = hi_vocab["<pad>"]

    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=6,
                        pin_memory=True)

    model = Transformer(
        src_vocab_size=len(en_vocab), tgt_vocab_size=len(hi_vocab),
        d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        d_ff=d_ff, max_len=MAX_LEN, dropout=dropout
    ).to(DEVICE)
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"  Config : lr={lr:.6f}, bs={batch_size}, heads={num_heads}, "
          f"d_ff={d_ff}, dropout={dropout:.3f}")
    print(f"  Epochs : {num_epochs}")
    print()

    loss_history = []
    bleu_history = []

    best_bleu = 0.0
    best_state = None
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = (src.to(DEVICE), tgt_in.to(DEVICE),
                                    tgt_out.to(DEVICE))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(src, tgt_in, SRC_PAD_IDX, TGT_PAD_IDX)
                loss = criterion(output.view(-1, output.shape[-1]),
                                 tgt_out.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            bleu = evaluate_bleu(model, en_vocab, hi_vocab, DEVICE)
            bleu_history.append((epoch + 1, bleu))
            if bleu > best_bleu:
                best_bleu = bleu
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  --  "
                  f"Loss: {avg_loss:.4f}  BLEU: {bleu:.4f}  "
                  f"(best: {best_bleu:.4f})")
        elif (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  --  Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time

    final_bleu = evaluate_bleu(model, en_vocab, hi_vocab, DEVICE)
    if final_bleu > best_bleu:
        best_bleu = final_bleu
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model_path = os.path.join(args.output_dir, "m25csa023_ass_4_best_model.pth")
    torch.save(best_state, model_path)

    final_metrics = {
        "time_sec": round(elapsed, 2),
        "time_min": round(elapsed / 60, 2),
        "final_loss": round(avg_loss, 4),
        "best_bleu": round(best_bleu, 4),
        "epochs": num_epochs,
        "config": best_config,
        "loss_history": [float(x) for x in loss_history],
        "bleu_history": [(int(e), float(b)) for e, b in bleu_history],
    }

    with open(os.path.join(args.output_dir, "tuned_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n  -- Final Tuned Results --")
    print(f"  Time       : {final_metrics['time_min']:.1f} min")
    print(f"  Final Loss : {final_metrics['final_loss']:.4f}")
    print(f"  Best BLEU  : {final_metrics['best_bleu']:.4f}  "
          f"({final_metrics['best_bleu']*100:.2f}%)")
    print(f"  Saved      : {model_path}")

    model.load_state_dict(best_state)
    print("\n  -- Sample Tuned Translations --")
    for en, hi in val_dataset:
        pred = translate_sentence(model, en, en_vocab, hi_vocab, DEVICE)
        print(f"    EN : {en}")
        print(f"    REF: {hi}")
        print(f"    HYP: {pred}")
        print()

    return final_metrics


#  PLOTTING 
def generate_all_plots(baseline, tuned, trial_data, output_dir):
    """
    Generate 9 plots total:
      1.  Baseline Training Loss Curve
      2.  Tuned Model Training Loss Curve
      3.  Baseline vs Tuned Loss Overlay
      4.  BLEU Score Progression Comparison
      5.  Ray Tune All Trial Loss Curves
      6a. Hyperparameter vs Final Loss (scatter subplots)
      6b. Hyperparameter Importance Ranking (Spearman correlation bar chart)
      7.  Final Metrics Bar Chart Comparison
      0.  Combined Summary Figure (all key visuals in one)
    """
    print("\n" + "=" * 70)
    print("  GENERATING PLOTS")
    print("=" * 70)

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    bl_loss = baseline.get("loss_history", [])
    bl_bleu = baseline.get("bleu_history", [])
    tu_loss = tuned.get("loss_history", [])
    tu_bleu = tuned.get("bleu_history", [])

    #  Baseline Training Loss Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(bl_loss) + 1)
    ax.plot(epochs, bl_loss, color=COLORS["baseline"], linewidth=1.8,
            label="Baseline Loss")
    ax.fill_between(epochs, bl_loss, alpha=0.08, color=COLORS["baseline"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Plot 1: Baseline Training Loss Curve (100 Epochs)")
    ax.legend(loc="upper right")
    if bl_loss:
        ax.set_xlim(1, len(bl_loss))
    fig.savefig(os.path.join(plots_dir, "1_baseline_loss_curve.png"))
    plt.close(fig)
    print("  [1/8] Baseline loss curve")

    # Tuned Model Training Loss Curve 
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_t = range(1, len(tu_loss) + 1)
    ax.plot(epochs_t, tu_loss, color=COLORS["tuned"], linewidth=1.8,
            label="Tuned Loss")
    ax.fill_between(epochs_t, tu_loss, alpha=0.08, color=COLORS["tuned"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Plot 2: Tuned Model Training Loss Curve "
                 f"({len(tu_loss)} Epochs)")
    ax.legend(loc="upper right")
    if tu_loss:
        ax.set_xlim(1, len(tu_loss))
    fig.savefig(os.path.join(plots_dir, "2_tuned_loss_curve.png"))
    plt.close(fig)
    print("  [2/8] Tuned loss curve")

    #  Baseline vs Tuned Loss Overlay 
    fig, ax = plt.subplots(figsize=(10, 5))
    if bl_loss:
        ax.plot(range(1, len(bl_loss)+1), bl_loss, color=COLORS["baseline"],
                linewidth=1.8, label=f"Baseline ({len(bl_loss)} epochs)",
                alpha=0.85)
    if tu_loss:
        ax.plot(range(1, len(tu_loss)+1), tu_loss, color=COLORS["tuned"],
                linewidth=2.2, label=f"Tuned ({len(tu_loss)} epochs)",
                linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Plot 3: Baseline vs Tuned - Training Loss Comparison")
    ax.legend(loc="upper right")
    if bl_loss or tu_loss:
        ax.set_xlim(1, max(len(bl_loss), len(tu_loss), 1))
    if tu_loss:
        ax.annotate(f"Tuned final: {tu_loss[-1]:.3f}",
                    xy=(len(tu_loss), tu_loss[-1]),
                    xytext=(max(1, len(tu_loss)-10),
                            tu_loss[-1] + max(0.1, (max(tu_loss)-min(tu_loss))*0.15)),
                    arrowprops=dict(arrowstyle="->", color=COLORS["tuned"]),
                    fontsize=9, color=COLORS["tuned"])
    fig.savefig(os.path.join(plots_dir, "3_baseline_vs_tuned_loss.png"))
    plt.close(fig)
    print("  [3/8] Loss overlay")

    #  BLEU Score Progression 
    fig, ax = plt.subplots(figsize=(10, 5))
    if bl_bleu:
        bl_ep, bl_sc = zip(*bl_bleu)
        ax.plot(bl_ep, bl_sc, color=COLORS["baseline"], linewidth=1.8,
                marker="o", markersize=6, label="Baseline BLEU")
        ax.axhline(y=bl_sc[-1], color=COLORS["baseline"], linestyle=":",
                   alpha=0.5, label=f"Baseline final: {bl_sc[-1]:.3f}")
    if tu_bleu:
        tu_ep, tu_sc = zip(*tu_bleu)
        ax.plot(tu_ep, tu_sc, color=COLORS["tuned"], linewidth=1.8,
                marker="s", markersize=6, label="Tuned BLEU")
    ax.axhline(y=0.50, color="#E91E63", linestyle="--", linewidth=1,
               alpha=0.6, label="Target BLEU (0.50)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU Score")
    ax.set_title("Plot 4: BLEU Score Progression - Baseline vs Tuned")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)
    fig.savefig(os.path.join(plots_dir, "4_bleu_comparison.png"))
    plt.close(fig)
    print("  [4/8] BLEU comparison")

    #  Ray Tune All Trial Loss Curves 
    fig, ax = plt.subplots(figsize=(12, 6))
    n_trials = len(trial_data)
    if trial_data:
        min_loss = min(t["final_loss"] for t in trial_data)
        max_epochs_seen = max(len(t.get("loss_curve", [])) for t in trial_data)
        for i, trial in enumerate(trial_data):
            lc = trial.get("loss_curve", [])
            if not lc:
                continue
            is_best = abs(trial["final_loss"] - min_loss) < 1e-6
            if is_best:
                ax.plot(range(1, len(lc)+1), lc, color=COLORS["tuned"],
                        linewidth=2.5, alpha=1.0, zorder=5,
                        label=f"Best trial (loss={trial['final_loss']:.3f})")
            else:
                ax.plot(range(1, len(lc)+1), lc, color="#999999",
                        linewidth=0.8, alpha=0.35)
        pruned = sum(1 for t in trial_data
                     if len(t.get("loss_curve", [])) < max_epochs_seen)
        ax.text(0.02, 0.02, f"Pruned by ASHA: {pruned}/{n_trials} trials",
                transform=ax.transAxes, fontsize=9, color="#666666",
                verticalalignment="bottom")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Plot 5: Ray Tune - All {n_trials} Trial Loss Curves")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(plots_dir, "5_ray_tune_all_trials.png"))
    plt.close(fig)
    print("  [5/8] Trial curves")

    #  Hyperparameter vs Final Loss (5 scatter subplots)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    fig.suptitle("Plot 6: Hyperparameter vs Final Loss "
                 "(color = loss, green=low, red=high)",
                 fontsize=13, fontweight="bold", y=1.03)

    if trial_data:
        losses = np.array([t["final_loss"] for t in trial_data])
        vmin, vmax = losses.min(), losses.max()
        if vmax == vmin:
            vmax = vmin + 1

        hp_keys =   ["lr", "batch_size", "num_heads", "d_ff", "dropout"]
        hp_labels = ["Learning Rate", "Batch Size", "Num Heads",
                     "FFN Dim (d_ff)", "Dropout Rate"]

        for ax_i, (hp, label) in enumerate(zip(hp_keys, hp_labels)):
            ax = axes[ax_i]
            vals = [t[hp] for t in trial_data]
            sc = ax.scatter(vals, losses, c=losses, cmap="RdYlGn_r",
                            vmin=vmin, vmax=vmax, s=70, edgecolors="white",
                            linewidths=0.6, zorder=3)
            ax.set_xlabel(label, fontsize=9)
            if ax_i == 0:
                ax.set_ylabel("Final Loss")
                ax.set_xscale("log")
            ax.tick_params(labelsize=8)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                    norm=plt.Normalize(vmin, vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("Final Loss", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "6a_hyperparameter_scatter.png"))
    plt.close(fig)
    print("  [6/9] HP scatter analysis")

    #  Hyperparameter Importance (Ranked Bar Chart) 
    #
    # Method: Spearman rank correlation |rho| between each HP and final loss.
    # This tells you which HP has the strongest monotonic relationship with
    # the loss — i.e., which HP matters most for convergence.
    # We also note that Optuna internally uses fANOVA for importance, but
    # that requires the Optuna study object. Here we compute a robust proxy.
    #
    fig, ax = plt.subplots(figsize=(9, 5))

    if trial_data and len(trial_data) >= 5:
        from scipy import stats as sp_stats

        hp_keys =   ["lr", "batch_size", "num_heads", "d_ff", "dropout"]
        hp_labels = ["Learning Rate", "Batch Size", "Num Heads",
                     "FFN Dim (d_ff)", "Dropout"]
        losses = np.array([t["final_loss"] for t in trial_data])

        importances = []
        p_values = []
        for hp in hp_keys:
            vals = np.array([float(t[hp]) for t in trial_data])
            rho, p = sp_stats.spearmanr(vals, losses)
            importances.append(abs(rho))
            p_values.append(p)

        # Sort by importance (descending)
        order = np.argsort(importances)[::-1]
        sorted_labels = [hp_labels[i] for i in order]
        sorted_imps = [importances[i] for i in order]
        sorted_pvals = [p_values[i] for i in order]

        # Color: green if significant (p<0.1), orange if borderline, grey if not
        bar_colors = []
        for p in sorted_pvals:
            if p < 0.05:
                bar_colors.append(COLORS["accent1"])   # Green — significant
            elif p < 0.15:
                bar_colors.append(COLORS["accent3"])   # Orange — borderline
            else:
                bar_colors.append("#BDBDBD")            # Grey — not significant

        y_pos = np.arange(len(sorted_labels))
        bars = ax.barh(y_pos, sorted_imps, color=bar_colors,
                       edgecolor="white", linewidth=1.2, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels, fontsize=11)
        ax.set_xlabel("|Spearman Correlation| with Final Loss", fontsize=11)
        ax.set_title("Plot 6b: Hyperparameter Importance Ranking",
                     fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()  # highest importance on top

        # Annotate bars with correlation value and significance
        for i, (bar, imp, p) in enumerate(zip(bars, sorted_imps, sorted_pvals)):
            sig_marker = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.15 else ""
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{imp:.3f} {sig_marker}",
                    va="center", fontsize=10, fontweight="bold")

        # Legend for significance
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS["accent1"], label="Significant (p < 0.05)"),
            Patch(facecolor=COLORS["accent3"], label="Borderline (p < 0.15)"),
            Patch(facecolor="#BDBDBD", label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        # Print importance ranking to console
        print("\n  ── Hyperparameter Importance Ranking ──")
        print(f"  {'Rank':<5s} {'Hyperparameter':<18s} {'|rho|':>8s} {'p-value':>10s} {'Significance':>14s}")
        for rank, i in enumerate(order, 1):
            sig = "***" if p_values[i] < 0.01 else "**" if p_values[i] < 0.05 else "*" if p_values[i] < 0.15 else ""
            print(f"  {rank:<5d} {hp_labels[i]:<18s} {importances[i]:>8.3f} {p_values[i]:>10.4f} {sig:>14s}")
        print()
    else:
        ax.text(0.5, 0.5, "Not enough trial data\n(need >= 5 trials)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Plot 6b: Hyperparameter Importance Ranking")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "6b_hyperparameter_importance.png"))
    plt.close(fig)
    print("  [7/9] HP importance ranking")

    #  Final Metrics Bar Chart 
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Plot 7: Baseline vs Tuned - Final Comparison",
                 fontsize=14, fontweight="bold")

    labels = ["Baseline", "Tuned"]
    x = np.arange(len(labels))

    # 7a: Training Time
    ax = axes[0]
    times = [baseline.get("time_min", 0), tuned.get("time_min", 0)]
    bars = ax.bar(x, times, 0.5,
                  color=[COLORS["baseline"], COLORS["tuned"]],
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Training Time")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    # 7b: Final Loss
    ax = axes[1]
    losses_cmp = [baseline.get("final_loss", 0), tuned.get("final_loss", 0)]
    bars = ax.bar(x, losses_cmp, 0.5,
                  color=[COLORS["baseline"], COLORS["tuned"]],
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Training Loss")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for bar, val in zip(bars, losses_cmp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    # 7c: BLEU Score
    ax = axes[2]
    bleus = [baseline.get("bleu_score", 0), tuned.get("best_bleu", 0)]
    bars = ax.bar(x, bleus, 0.5,
                  color=[COLORS["baseline"], COLORS["tuned"]],
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("BLEU Score")
    ax.set_title("BLEU Score")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(y=0.50, color="#E91E63", linestyle="--", linewidth=1.2,
               label="Target (0.50)")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, bleus):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "7_final_comparison_bars.png"))
    plt.close(fig)
    print("  [8/9] Final bar chart")

    # ─── Plot 0: Combined Summary Figure (2 rows x 4 cols) ───────────────
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle("Assignment 4 - Transformer EN->HI "
                 "Hyperparameter Tuning Summary",
                 fontsize=16, fontweight="bold", y=0.98)

    # (a) Loss overlay
    ax = fig.add_subplot(gs[0, 0])
    if bl_loss:
        ax.plot(range(1, len(bl_loss)+1), bl_loss, color=COLORS["baseline"],
                linewidth=1.5, label=f"Baseline ({len(bl_loss)}ep)")
    if tu_loss:
        ax.plot(range(1, len(tu_loss)+1), tu_loss, color=COLORS["tuned"],
                linewidth=1.5, linestyle="--",
                label=f"Tuned ({len(tu_loss)}ep)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("(a) Training Loss"); ax.legend(fontsize=8)

    # (b) BLEU comparison
    ax = fig.add_subplot(gs[0, 1])
    if bl_bleu:
        ax.plot(*zip(*bl_bleu), color=COLORS["baseline"], marker="o",
                markersize=4, linewidth=1.5, label="Baseline")
    if tu_bleu:
        ax.plot(*zip(*tu_bleu), color=COLORS["tuned"], marker="s",
                markersize=4, linewidth=1.5, label="Tuned")
    ax.axhline(y=0.50, color="#E91E63", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Target")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BLEU")
    ax.set_title("(b) BLEU Progression"); ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # (c) All trial curves
    ax = fig.add_subplot(gs[0, 2])
    if trial_data:
        min_loss_val = min(t["final_loss"] for t in trial_data)
        for trial in trial_data:
            lc = trial.get("loss_curve", [])
            if lc:
                is_best = abs(trial["final_loss"] - min_loss_val) < 1e-6
                ax.plot(range(1, len(lc)+1), lc,
                        color=COLORS["tuned"] if is_best else "#999999",
                        alpha=1.0 if is_best else 0.3,
                        linewidth=2.0 if is_best else 0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"(c) All {len(trial_data)} Tune Trials")

    # (d) HP Importance — mini version of Plot 6b
    ax = fig.add_subplot(gs[0, 3])
    if trial_data and len(trial_data) >= 5:
        from scipy import stats as sp_stats
        hp_keys_s =   ["lr", "batch_size", "num_heads", "d_ff", "dropout"]
        hp_labels_s = ["LR", "Batch", "Heads", "FFN", "Dropout"]
        losses_s = np.array([t["final_loss"] for t in trial_data])
        imps = []
        for hp in hp_keys_s:
            vals = np.array([float(t[hp]) for t in trial_data])
            rho, _ = sp_stats.spearmanr(vals, losses_s)
            imps.append(abs(rho))
        order_s = np.argsort(imps)[::-1]
        y_pos_s = np.arange(len(hp_labels_s))
        colors_s = [COLORS["accent1"] if imps[i] > 0.3 else
                     COLORS["accent3"] if imps[i] > 0.15 else "#BDBDBD"
                     for i in order_s]
        ax.barh(y_pos_s, [imps[i] for i in order_s], color=colors_s,
                edgecolor="white", height=0.6)
        ax.set_yticks(y_pos_s)
        ax.set_yticklabels([hp_labels_s[i] for i in order_s], fontsize=9)
        ax.set_xlabel("|Spearman rho|", fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_title("(d) HP Importance")

    # (e) LR vs Loss
    ax = fig.add_subplot(gs[1, 0])
    if trial_data:
        lrs = [t["lr"] for t in trial_data]
        fl = [t["final_loss"] for t in trial_data]
        sc = ax.scatter(lrs, fl, c=fl, cmap="RdYlGn_r", s=80,
                        edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log)"); ax.set_ylabel("Final Loss")
        ax.set_title("(e) LR vs Final Loss")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    # (f) Dropout vs Loss
    ax = fig.add_subplot(gs[1, 1])
    if trial_data:
        drs = [t["dropout"] for t in trial_data]
        fl = [t["final_loss"] for t in trial_data]
        sc = ax.scatter(drs, fl, c=fl, cmap="RdYlGn_r", s=80,
                        edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_xlabel("Dropout Rate"); ax.set_ylabel("Final Loss")
        ax.set_title("(f) Dropout vs Final Loss")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    # (g) Batch Size vs Loss (box-style scatter)
    ax = fig.add_subplot(gs[1, 2])
    if trial_data:
        bs_vals = [t["batch_size"] for t in trial_data]
        fl = [t["final_loss"] for t in trial_data]
        sc = ax.scatter(bs_vals, fl, c=fl, cmap="RdYlGn_r", s=80,
                        edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_xlabel("Batch Size"); ax.set_ylabel("Final Loss")
        ax.set_title("(g) Batch Size vs Final Loss")

    # (h) Final comparison bars
    ax = fig.add_subplot(gs[1, 3])
    metrics_names = ["Time\n(min)", "Loss", "BLEU"]
    bl_vals = [baseline.get("time_min", 0), baseline.get("final_loss", 0),
               baseline.get("bleu_score", 0)]
    tu_vals = [tuned.get("time_min", 0), tuned.get("final_loss", 0),
               tuned.get("best_bleu", 0)]
    x_pos = np.arange(len(metrics_names))
    w = 0.3
    ax.bar(x_pos - w/2, bl_vals, w, color=COLORS["baseline"],
           label="Baseline", edgecolor="white")
    ax.bar(x_pos + w/2, tu_vals, w, color=COLORS["tuned"],
           label="Tuned", edgecolor="white")
    ax.set_xticks(x_pos); ax.set_xticklabels(metrics_names)
    ax.set_title("(h) Final Comparison"); ax.legend(fontsize=8)

    fig.savefig(os.path.join(plots_dir, "0_summary_figure.png"))
    plt.close(fig)
    print("  [9/9] Combined summary figure")

    print(f"\n  All 9 plots saved to: {plots_dir}/")
    for fname in sorted(os.listdir(plots_dir)):
        print(f"    - {fname}")


# 8.  MAIN
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    df, en_vocab, hi_vocab = load_data(args.data_path)

    # ---- Part 1: Baseline ----
    if not args.skip_baseline:
        baseline = run_baseline(df, en_vocab, hi_vocab, args.output_dir,
                                num_epochs=100)
    else:
        print("\n[INFO] Skipping baseline (--skip_baseline). "
              "Loading from baseline_metrics.json.\n")
        bp = os.path.join(args.output_dir, "baseline_metrics.json")
        if os.path.exists(bp):
            with open(bp) as f:
                baseline = json.load(f)
            print(f"  Loaded baseline: BLEU={baseline.get('bleu_score')}, "
                  f"Loss={baseline.get('final_loss')}")
        else:
            baseline = {"bleu_score": 0.50, "final_loss": 999, "time_sec": 0,
                        "time_min": 0, "epochs": 100,
                        "loss_history": [], "bleu_history": []}

    #  Ray Tune + Optuna sweep 
    ray.init(ignore_reinit_error=True)
    best_config, trial_data = run_tuning(df, en_vocab, hi_vocab, args)
    ray.shutdown()

    #  Retrain best & save 
    tuned = retrain_best(df, en_vocab, hi_vocab, best_config, args)

    #  Generate all plots 
    generate_all_plots(baseline, tuned, trial_data, args.output_dir)

    #  Summary 

    print(f"  {'Metric':<20s} {'Baseline':>12s} {'Tuned':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'Epochs':<20s} {baseline.get('epochs', 100):>12} "
          f"{tuned['epochs']:>12d}")
    print(f"  {'Time (min)':<20s} {baseline.get('time_min', '?'):>12} "
          f"{tuned['time_min']:>12.1f}")
    print(f"  {'Final Loss':<20s} {baseline.get('final_loss', '?'):>12} "
          f"{tuned['final_loss']:>12.4f}")
    bl_bleu_val = baseline.get('bleu_score', baseline.get('bleu', '?'))
    print(f"  {'BLEU Score':<20s} {bl_bleu_val:>12} "
          f"{tuned['best_bleu']:>12.4f}")
    print()
    print("  Files saved:")
    print(f"    - transformer_translation_final.pth  (baseline weights)")
    print(f"    - m25csa023_ass_4_best_model.pth     (best tuned weights)")
    print(f"    - baseline_metrics.json")
    print(f"    - tuned_metrics.json")
    print(f"    - best_config.json")
    print(f"    - trial_data.json")
    print(f"    - en_vocab.pkl, hi_vocab.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()