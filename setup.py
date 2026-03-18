"""
setup.py
--------
Install package + dependencies, then download models separately:

    pip install git+https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023.git@Assignment-4
    download-en-hi-models

The second command is registered as a console_script by this setup.py
and becomes available system-wide after pip install completes.
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
