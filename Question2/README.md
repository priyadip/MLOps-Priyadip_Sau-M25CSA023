# Question 2 — CityScape UNet Segmentation + Streamlit App

## Files
- `download_data.py` — pulls the dataset folder via `gdown`
- `dataset.py` — `CityscapesDataset` + path collector (same snippet shape as the exam)
- `unet_model.py` — UNet with 23 output channels
- `metrics.py` — confusion-matrix-based mIoU / mDice
- `train.py` — trains, logs plots to `plots/`, saves best checkpoint
- `evaluate.py` — runs test set only, writes `test_metrics.json`
- `app.py` — 2-page Streamlit app

## Steps
```bash
pip install -r requirements.txt

# 1. Download
python download_data.py

# 2. Train (>=15 epochs)
python train.py --data_dir data --epochs 20 --batch_size 16

# 3. Evaluate on test set
python evaluate.py --data_dir data --ckpt checkpoints/unet_best.pt

# 4. Launch app
streamlit run app.py
```

## Pushing plots to GitHub (required for marks)
```bash
git checkout MLDLOPs-Exam2026
git add Question2/plots/train_loss.png Question2/plots/miou.png Question2/plots/mdice.png
git commit -m "Question2: add training loss, mIoU, mDice plots"
git push origin MLDLOPs-Exam2026
```

## Required README.md line (at repo root)
After `evaluate.py` prints the numbers, run:
```bash
echo "Question2: mIOU: 0.XXXX and mDICE: 0.XXXX" >> README.md
git add README.md
git commit -m "Question2: add test mIoU and mDice to README"
git push origin MLDLOPs-Exam2026
```
