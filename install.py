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

import os
import sys
import urllib.request

REPO_URL    = "https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023.git"
BRANCH      = "Assignment-4"
CLONE_DIR   = os.path.join(os.getcwd(), "MLOps-Assignment-4")
HF_REPO_ID  = "priyadip/en-hi-transformer"

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

# Step 1: Download only the files we want via GitHub API -------------------
import json

GH_API  = f"https://api.github.com/repos/priyadip/MLOps-Priyadip_Sau-M25CSA023"
RAW_URL = f"https://raw.githubusercontent.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/{BRANCH}"

# Only these paths will be downloaded from the repo
INCLUDE_PATHS = [
    "m25csa023_ass_4_tuned_en_to_hi.py",
    "results",
    "report",
]


def gh_list_files(api_path):
    """Return list of (download_url, relative_path) for all files under api_path."""
    url = f"{GH_API}/contents/{api_path}?ref={BRANCH}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    data = json.loads(urllib.request.urlopen(req).read())
    files = []
    for item in data:
        if item["type"] == "file":
            files.append((item["download_url"], item["path"]))
        elif item["type"] == "dir":
            files.extend(gh_list_files(item["path"]))
    return files


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"  [SKIP]  {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  [DOWN]  {dest}")
    urllib.request.urlretrieve(url, dest)


print(f"\n[1/2]  Downloading repo files into {CLONE_DIR} ...")
os.makedirs(CLONE_DIR, exist_ok=True)
for path in INCLUDE_PATHS:
    try:
        for dl_url, rel_path in gh_list_files(path):
            download_file(dl_url, os.path.join(CLONE_DIR, rel_path))
    except Exception:
        # single file (not a folder)
        download_file(f"{RAW_URL}/{path}", os.path.join(CLONE_DIR, path))
print(f"  [ OK ]  Files ready at {CLONE_DIR}")

# Step 2: Download model weights -------------------------------------------
print("\n[2/2]  Downloading model weights from Hugging Face ...")
print(f"  Repo : https://huggingface.co/{HF_REPO_ID}")

for m in MODELS:
    print(f"\n  -- {m['version']}  ({m['label']})")
    download_model_file(m)

print("\n" + "=" * 60)
print(f"  Done! Everything is ready in: {CLONE_DIR}")
print("  - Code + results: see subfolders")
print("  - Run the model:  python m25csa023_ass_4_tuned_en_to_hi.py")
print("=" * 60 + "\n")
