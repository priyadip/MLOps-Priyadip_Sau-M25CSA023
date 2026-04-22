"""
Q1 

input.txt   (one Bengali sentence per line)
output.txt  (one English sentence per line, same order)
"""

from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"
INPUT_FILE = Path("input.txt")
OUTPUT_FILE = Path("output.txt")
BATCH_SIZE = 8


def load_model(device: torch.device):
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model


def translate_batch(lines, tokenizer, model, device):
    enc = tokenizer(lines, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        gen = model.generate(**enc, max_length=256, num_beams=4)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    tokenizer, model = load_model(device)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    print(f"[info] {len(lines)} input sentences")

    outputs = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i : i + BATCH_SIZE]
        outputs.extend(translate_batch(batch, tokenizer, model, device))
        print(f"[info] translated {min(i + BATCH_SIZE, len(lines))}/{len(lines)}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for t in outputs:
            f.write(t.strip() + "\n")

    print(f"[done] wrote {OUTPUT_FILE.resolve()}")
    if outputs:
        print("\n=== First statement translation ===")
        print(outputs[0])


if __name__ == "__main__":
    main()
