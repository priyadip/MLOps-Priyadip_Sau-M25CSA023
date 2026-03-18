"""
install.py  —  bootstrap installer
-----------------------------------
One command, no pre-install needed:

  python -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/Assignment-4/install.py').read())"

What this does:
  1. pip-installs all dependencies (torch, huggingface_hub, ray, optuna, nltk)
  2. Downloads both EN→HI model checkpoints from Hugging Face
     into your current working directory
"""

import subprocess
import sys
import os
import urllib.request
import tempfile

DEPS = ["torch>=2.0", "huggingface_hub>=0.20", "nltk", "ray[tune]", "optuna"]
DL_URL = (
    "https://raw.githubusercontent.com/priyadip/"
    "MLOps-Priyadip_Sau-M25CSA023/Assignment-4/download_model.py"
)


def run(cmd, **kw):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    subprocess.check_call(cmd, **kw)


print("=" * 60)
print("  EN→HI Transformer  —  one-command installer")
print("=" * 60)

# ── Step 1: install dependencies ─────────────────────────────────────────────
print("\n[1/2]  Installing dependencies ...")
run([sys.executable, "-m", "pip", "install"] + DEPS)

# ── Step 2: download model weights ───────────────────────────────────────────
print("\n[2/2]  Downloading model weights from Hugging Face ...")
with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
    f.write(urllib.request.urlopen(DL_URL).read())
    tmp = f.name

try:
    run([sys.executable, tmp], cwd=os.getcwd())
finally:
    os.unlink(tmp)

print("\n" + "=" * 60)
print("  Done. Models are in your current folder.")
print("=" * 60 + "\n")
