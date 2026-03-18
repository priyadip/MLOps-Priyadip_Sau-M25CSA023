"""
setup.py
--------
Installs the package and auto-downloads model weights from Hugging Face.

    pip install .
    pip install git+https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023.git@Assignment-4

    # Skip model download (CI / custom builds):
    SKIP_MODEL_DOWNLOAD=1 pip install .
"""

import os
import subprocess
import sys
from setuptools import setup, find_packages


def download_models():
    if os.getenv("SKIP_MODEL_DOWNLOAD") == "1":
        print("[setup.py] SKIP_MODEL_DOWNLOAD=1 — skipping.")
        return
    try:
        print("[setup.py] Downloading model weights from Hugging Face...")
        subprocess.check_call([sys.executable, "download_model.py"])
    except Exception as exc:
        print(f"[setup.py] Model download failed: {exc}")
        print("[setup.py] Run manually:  python download_model.py")


download_models()

setup(
    name="en_hi_transformer",
    version="1.1.0",
    description="English → Hindi Transformer (from-scratch PyTorch, Ray Tune optimised)",
    author="Priyadip Sau",
    author_email="m25csa023@iitj.ac.in",
    url="https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/tree/Assignment-4",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "huggingface_hub>=0.20",
        "nltk",
        "ray[tune]",
        "optuna",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
