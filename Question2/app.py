"""
Q2 — Streamlit 2-page deployment app for the trained UNet.

Pages:
    1. Training curves + test-set mIoU / mDice
    2. Upload 4 test images; show GT mask (if paired) + predicted mask

Run:
    streamlit run app.py
"""

import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from dataset import IMG_H, IMG_W
from unet_model import UNet


CKPT_PATH = "checkpoints/unet_best.pt"
PLOTS_DIR = "plots"
TEST_METRICS_PATH = "test_metrics.json"
N_CLASSES = 23


# ---------- color palette for 23-class visualization -------------------------
def build_palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pal = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    pal[0] = [0, 0, 0]  # class 0 → black
    return pal


PALETTE = build_palette(N_CLASSES)


def colorize(mask_2d: np.ndarray) -> np.ndarray:
    mask_2d = np.clip(mask_2d, 0, N_CLASSES - 1).astype(np.int64)
    return PALETTE[mask_2d]


# ---------- model loading (cached) -------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=N_CLASSES).to(device)
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


# ---------- inference helpers ------------------------------------------------
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"))
    arr = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def predict_mask(model, device, pil_img: Image.Image) -> np.ndarray:
    x = preprocess(pil_img).to(device)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(dim=1)[0].cpu().numpy()


def derive_gt_mask(pil_img: Image.Image) -> np.ndarray:
    """Same collapse as dataset.py: np.max over RGB channels."""
    arr = np.array(pil_img.convert("RGB"))
    arr = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    return np.max(arr, axis=-1)


# =============================================================================
st.set_page_config(page_title="CityScape Segmentation", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Training Metrics", "🖼 Upload & Segment"])

model, device = load_model()

# -------------------- Page 1 -------------------------------------------------
if page == "📊 Training Metrics":
    st.title("CityScape UNet — Training Metrics")
    st.caption(f"Device: {device}")

    cols = st.columns(3)
    for col, name, title in zip(
        cols,
        ["train_loss.png", "miou.png", "mdice.png"],
        ["Training Loss", "mIoU", "mDice"],
    ):
        with col:
            st.subheader(title)
            p = Path(PLOTS_DIR) / name
            if p.exists():
                st.image(str(p), use_column_width=True)
            else:
                st.warning(f"Plot not found: {p} (run train.py first)")

    st.subheader("Test-Set Metrics")
    if Path(TEST_METRICS_PATH).exists():
        m = json.loads(Path(TEST_METRICS_PATH).read_text())
        c1, c2 = st.columns(2)
        c1.metric("Test mIoU", f"{m['mIoU']:.4f}")
        c2.metric("Test mDice", f"{m['mDice']:.4f}")
    else:
        st.info("Run `python evaluate.py` to populate test_metrics.json.")

# -------------------- Page 2 -------------------------------------------------
else:
    st.title("Upload 4 Test Images — Predicted vs Ground-Truth Masks")
    st.caption(
        "Upload 4 RGB images from CameraRGB *and* their paired masks from CameraMask. "
        "If you only upload RGB images, only the predicted mask is shown."
    )

    c1, c2 = st.columns(2)
    rgb_files = c1.file_uploader(
        "RGB images (4)", accept_multiple_files=True,
        type=["png", "jpg", "jpeg"], key="rgb",
    )
    mask_files = c2.file_uploader(
        "Ground-truth masks (4, optional)", accept_multiple_files=True,
        type=["png", "jpg", "jpeg"], key="mask",
    )

    if rgb_files:
        rgb_sorted = sorted(rgb_files, key=lambda f: f.name)[:4]
        mask_map = {f.name: f for f in (mask_files or [])}

        for i, f in enumerate(rgb_sorted, 1):
            img = Image.open(io.BytesIO(f.read()))
            pred = predict_mask(model, device, img)
            pred_vis = colorize(pred)

            st.markdown(f"### Sample {i} — `{f.name}`")
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Input (RGB)", use_column_width=True)

            if f.name in mask_map:
                gt_img = Image.open(io.BytesIO(mask_map[f.name].read()))
                gt = derive_gt_mask(gt_img)
                c2.image(colorize(gt), caption="Ground Truth", use_column_width=True)
            else:
                c2.info("No matching GT mask uploaded.")

            c3.image(pred_vis, caption="Predicted Mask", use_column_width=True)
