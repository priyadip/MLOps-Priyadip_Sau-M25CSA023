"""
Q2 — download and extract CityScape dataset.

The question points to a Google Drive folder:
    https://drive.google.com/drive/u/1/folders/1GNe3Tu8Mud_CSLOiQZYHS2Rjq2sS74b_

Run from inside Question2/:
    python download_data.py
"""

import os
import subprocess
import sys

FOLDER_ID = "1GNe3Tu8Mud_CSLOiQZYHS2Rjq2sS74b_"
TARGET_DIR = "data"


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    # `gdown --folder` downloads an entire Google Drive folder.
    cmd = [
        sys.executable, "-m", "gdown",
        f"https://drive.google.com/drive/folders/{FOLDER_ID}",
        "--folder",
        "-O", TARGET_DIR,
    ]
    print("[info] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # If a zip was downloaded, unzip it
    for f in os.listdir(TARGET_DIR):
        if f.lower().endswith(".zip"):
            print(f"[info] unzipping {f}")
            subprocess.run(["unzip", "-o", os.path.join(TARGET_DIR, f), "-d", TARGET_DIR], check=True)

    print("[done] data in:", os.path.abspath(TARGET_DIR))


if __name__ == "__main__":
    main()
