"""
setup.py
--------
Single command — works on PowerShell, CMD, and bash:

    pip install git+https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023.git@Assignment-4

What happens automatically:
  1. GitHub repo is cloned
  2. Dependencies are installed (torch, ray, optuna, huggingface_hub ...)
  3. Model weights are downloaded from Hugging Face into your current folder
  4. Package is ready to use

Skip model download (CI / offline):
    SKIP_MODEL_DOWNLOAD=1 pip install git+https://...
"""

import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop


def _download():
    if os.getenv("SKIP_MODEL_DOWNLOAD") == "1":
        print("[setup] SKIP_MODEL_DOWNLOAD=1 — skipping model download.")
        return
    print("[setup] Downloading model weights from Hugging Face ...")
    try:
        # download_model is packaged as a py_module — run it directly
        subprocess.check_call(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "import download_model; download_model.main()"],
            cwd=os.getcwd(),          # user's working directory, not temp
        )
    except Exception as exc:
        print(f"[setup] Download failed: {exc}")
        print("[setup] Run manually after install:  download-en-hi-models")


class PostInstall(install):
    """Trigger model download right after pip installs the package."""
    def run(self):
        install.run(self)
        _download()


class PostDevelop(develop):
    """Trigger model download for editable installs (pip install -e .)."""
    def run(self):
        develop.run(self)
        _download()


setup(
    name="en_hi_transformer",
    version="1.1.0",
    description="English → Hindi Transformer (from-scratch PyTorch, Ray Tune optimised)",
    author="Priyadip Sau",
    author_email="m25csa023@iitj.ac.in",
    url="https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/tree/Assignment-4",
    license="MIT",
    py_modules=["download_model"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "huggingface_hub>=0.20",
        "nltk",
        "ray[tune]",
        "optuna",
    ],
    cmdclass={
        "install": PostInstall,
        "develop": PostDevelop,
    },
    entry_points={
        "console_scripts": [
            # fallback: run this if auto-download didn't trigger
            "download-en-hi-models=download_model:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
