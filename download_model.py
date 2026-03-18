"""
download_model.py
-----------------
Downloads both EN→HI Transformer checkpoints from Hugging Face Hub.

Usage (standalone):
    python download_model.py

Controlled via env var (skip during CI / custom builds):
    SKIP_MODEL_DOWNLOAD=1 python download_model.py
"""

import os
import sys

HF_REPO_ID   = "priyadip/en-hi-transformer"
V100_DIR     = "./transformer_translation_final"   # v1.0.0 baseline
V110_DIR     = "./rollno_ass_4_best_model"         # v1.1.0 optimised


def _require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        print("        Run:  pip install huggingface_hub")
        sys.exit(1)


def download_model(remote_path: str, local_dir: str, filename: str) -> None:
    hf_hub_download = _require_hf_hub()

    local_file = os.path.join(local_dir, filename)
    if os.path.exists(local_file):
        size_mb = os.path.getsize(local_file) / 1024 ** 2
        print(f"  [SKIP]  {local_file}  ({size_mb:.0f} MB already present)")
        return

    os.makedirs(local_dir, exist_ok=True)
    print(f"  [DOWN]  {HF_REPO_ID}/{remote_path}  →  {local_file}")
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            local_dir=local_dir,
        )
        size_mb = os.path.getsize(local_file) / 1024 ** 2
        print(f"  [ OK ]  {filename}  ({size_mb:.0f} MB)")
    except Exception as exc:
        print(f"  [FAIL]  {exc}")
        print(f"          Download manually from:")
        print(f"          https://huggingface.co/{HF_REPO_ID}/tree/main/{os.path.dirname(remote_path)}")


def main():
    if os.getenv("SKIP_MODEL_DOWNLOAD") == "1":
        print("[INFO] SKIP_MODEL_DOWNLOAD=1 — skipping model download.")
        return

    print("=" * 60)
    print("  Downloading EN→HI Transformer models from Hugging Face")
    print(f"  Repo : https://huggingface.co/{HF_REPO_ID}")
    print("=" * 60)

    # v1.0.0 — baseline
    print("\n── v1.0.0  (baseline, BLEU 0.7566, 100 epochs)")
    download_model(
        remote_path="v1.0.0/transformer_translation_final.pth",
        local_dir=V100_DIR,
        filename="transformer_translation_final.pth",
    )

    # v1.1.0 — Ray Tune optimised
    print("\n── v1.1.0  (optimised, BLEU 0.8369, 50 epochs)  ← recommended")
    download_model(
        remote_path="v1.1.0/m25csa023_ass_4_best_model.pth",
        local_dir=V110_DIR,
        filename="m25csa023_ass_4_best_model.pth",
    )

    print("\n" + "=" * 60)
    print("  Done. Models ready to use.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
