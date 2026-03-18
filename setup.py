"""
setup.py
--------
Install the package, then download models by running:

    download-en-hi-models

Or directly:
    python download_model.py

Or skip download (CI / custom builds):
    SKIP_MODEL_DOWNLOAD=1 pip install .
"""

from setuptools import setup

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
    entry_points={
        "console_scripts": [
            "download-en-hi-models=download_model:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
