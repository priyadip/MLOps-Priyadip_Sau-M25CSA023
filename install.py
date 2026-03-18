"""
install.py  —  one-command installer
--------------------------------------
Run this single command to get EVERYTHING:
  - All code files cloned from GitHub
  - All model weights downloaded from Hugging Face

One command (copy-paste this):

  python -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/Assignment-4/install.py').read())"

What this does:
  1. Clones the full GitHub repo (code, results, plots, report) into ./MLOps-Assignment-4/
  2. pip-installs all dependencies (torch, huggingface_hub, ray, optuna, nltk)
  3. Downloads both EN->HI model checkpoints from Hugging Face
     into ./MLOps-Assignment-4/transformer_translation_final/ and ./MLOps-Assignment-4/m25csa023_ass_4_best_model/
"""

import subprocess
import sys
import os

REPO_URL    = "https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023.git"
BRANCH      = "Assignment-4"
CLONE_DIR   = os.path.join(os.getcwd(), "MLOps-Assignment-4")
HF_REPO_ID  = "priyadip/en-hi-transformer"

DEPS = [
    "torch>=2.0",
    "huggingface_hub>=0.20",
    "nltk",
    "ray[tune]",
    "optuna",
]

MODELS = [
    {
        "version":  "v1.0.0",
        "label":    "baseline, BLEU 0.7566, 100 epochs",
        "url":      f"https://huggingface.co/{HF_REPO_ID}/resolve/main"
                    "/v1.0.0/transformer_translation_final.pth",
        "local_dir": CLONE_DIR,
        "filename":  "transformer_translation_final.pth",
    },
    {
        "version":  "v1.1.0",
        "label":    "optimised, BLEU 0.8369, 50 epochs  <- recommended",
        "url":      f"https://huggingface.co/{HF_REPO_ID}/resolve/main"
                    "/v1.1.0/m25csa023_ass_4_best_model.pth",
        "local_dir": CLONE_DIR,
        "filename":  "m25csa023_ass_4_best_model.pth",
    },
]


def run(cmd, **kw):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    subprocess.check_call(cmd, **kw)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        sys.stdout.write(f"\r  [{'#' * bar}{'.' * (50 - bar)}] {pct:5.1f}%")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()


def download_model_file(model):
    import urllib.request
    local_file = os.path.join(model["local_dir"], model["filename"])
    if os.path.exists(local_file):
        size_mb = os.path.getsize(local_file) / 1024 ** 2
        print(f"  [SKIP]  {local_file}  ({size_mb:.0f} MB already present)")
        return
    os.makedirs(model["local_dir"], exist_ok=True)
    print(f"  [DOWN]  {model['filename']}  ->  {local_file}")
    try:
        urllib.request.urlretrieve(model["url"], local_file, _progress)
        size_mb = os.path.getsize(local_file) / 1024 ** 2
        print(f"  [ OK ]  {model['filename']}  ({size_mb:.0f} MB)")
    except Exception as exc:
        print(f"  [FAIL]  {exc}")
        print(f"          Download manually: {model['url']}")
        if os.path.exists(local_file):
            os.remove(local_file)


print("=" * 60)
print("  EN->HI Transformer  --  one-command installer")
print("=" * 60)

# Step 1: Clone repo -------------------------------------------------------
print(f"\n[1/3]  Cloning GitHub repo into {CLONE_DIR} ...")
if os.path.exists(CLONE_DIR):
    print(f"  [SKIP]  {CLONE_DIR} already exists — pulling latest changes ...")
    run(["git", "-C", CLONE_DIR, "pull"])
else:
    run(["git", "clone", "-b", BRANCH, REPO_URL, CLONE_DIR])
print(f"  [ OK ]  Repo ready at {CLONE_DIR}")

# Step 2: Install dependencies ---------------------------------------------
print("\n[2/3]  Installing Python dependencies ...")
run([sys.executable, "-m", "pip", "install"] + DEPS)

# Step 3: Download model weights -------------------------------------------
print("\n[3/3]  Downloading model weights from Hugging Face ...")
print(f"  Repo : https://huggingface.co/{HF_REPO_ID}")

for m in MODELS:
    print(f"\n  -- {m['version']}  ({m['label']})")
    download_model_file(m)

print("\n" + "=" * 60)
print(f"  Done! Everything is ready in: {CLONE_DIR}")
print("  - Code + results: see subfolders")
print("  - Run the model:  python m25csa023_ass_4_tuned_en_to_hi.py")
print("=" * 60 + "\n")
