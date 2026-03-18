"""
download_model.py
-----------------
Downloads both EN→HI Transformer checkpoints directly from Hugging Face.
Uses only Python standard library — no extra packages needed.

Usage:
    python download_model.py
"""

import os
import sys
import urllib.request

MODELS = [
    {
        "version": "v1.0.0",
        "label":   "baseline, BLEU 0.7566, 100 epochs",
        "url":     "https://huggingface.co/priyadip/en-hi-transformer"
                   "/resolve/main/v1.0.0/transformer_translation_final.pth",
        "local_dir":  "./transformer_translation_final",
        "filename":   "transformer_translation_final.pth",
    },
    {
        "version": "v1.1.0",
        "label":   "optimised, BLEU 0.8369, 50 epochs  ← recommended",
        "url":     "https://huggingface.co/priyadip/en-hi-transformer"
                   "/resolve/main/v1.1.0/m25csa023_ass_4_best_model.pth",
        "local_dir":  "./m25csa023_ass_4_best_model",
        "filename":   "m25csa023_ass_4_best_model.pth",
    },
]


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        sys.stdout.write(f"\r  [{'█' * bar}{'░' * (50 - bar)}] {pct:5.1f}%")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()


def download(model: dict) -> None:
    local_file = os.path.join(model["local_dir"], model["filename"])

    if os.path.exists(local_file):
        size_mb = os.path.getsize(local_file) / 1024 ** 2
        print(f"  [SKIP]  already present  ({size_mb:.0f} MB)")
        return

    os.makedirs(model["local_dir"], exist_ok=True)
    print(f"  [DOWN]  {model['url'].split('resolve/main/')[-1]}"
          f"  →  {local_file}")
    try:
        urllib.request.urlretrieve(model["url"], local_file, _progress)
        size_mb = os.path.getsize(local_file) / 1024 ** 2
        print(f"  [ OK ]  {model['filename']}  ({size_mb:.0f} MB)")
    except Exception as exc:
        print(f"\n  [FAIL]  {exc}")
        print(f"          Download manually:")
        print(f"          {model['url']}")
        if os.path.exists(local_file):
            os.remove(local_file)   # remove partial download


def main():
    print("=" * 60)
    print("  Downloading EN→HI Transformer models from Hugging Face")
    print("  Repo : https://huggingface.co/priyadip/en-hi-transformer")
    print("=" * 60)

    for m in MODELS:
        print(f"\n── {m['version']}  ({m['label']})")
        download(m)

    print("\n" + "=" * 60)
    print("  Done. Models ready to use.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
