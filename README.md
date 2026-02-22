# Goodreads Genre Classifier — MLOps Assignment 3

**End-to-end ML pipeline:** Fine-tune **DistilBERT** on Goodreads book reviews for 8-class genre classification, containerize with Docker, and publish to Hugging Face Hub.

- **Hugging Face Model:** [priyadip/goodreads-genre-classifier](https://huggingface.co/priyadip/goodreads-genre-classifier)
- **GitHub Repo:** [https://github.com/priyadip/dlops-assignment3](https://github.com/priyadip/MLOps-Priyadip_Sau-M25CSA023/tree/Assignment-3)

---

## Project Structure

```
dlops-assignment3/
├── src/                        # Core library modules (Task 3)
│   ├── config.py               # All constants, hyperparams, paths
│   ├── data.py                 # Download, cache, train/test split
│   ├── dataset.py              # Custom PyTorch Dataset
│   ├── model.py                # Model & tokenizer loading
│   └── utils.py                # Metrics, reports, result I/O
├── scripts/                    # Executable entry points
│   ├── train.py                # Full training pipeline (Task 5)
│   ├── evaluate.py             # Evaluate local or Hub model (Task 6 & 8)
│   ├── push_to_hub.py          # Push to HF profile (Task 7)
│   └── eval_from_hub.py        # Re-evaluate from Hub + compare (Task 8)
├── slurm/
│   └── train.slurm             # SBATCH job for university GPU cluster
├── results/                    # Evaluation metrics & reports
│   ├── eval_local_post_train_20260217_081856.json
│   ├── eval_local_20260217_105009.json
│   ├── eval_hub_20260218_023325.json
│   ├── eval_hub_20260218_093412.json
│   ├── classification_report_local.txt
│   └── classification_report_hub.txt
├── Dockerfile                  # Dev/training image (Task 2)
├── Dockerfile.eval             # Production eval-only image (Task 9)
├── requirements.txt
└── README.md
```

---

## Model Selection Rationale

**Model:** `distilbert-base-cased`

| Criteria | Why DistilBERT-base-cased |
|----------|---------------------------|
| **Size** | 66M params — 40% smaller than BERT-base, fits comfortably in limited GPU memory |
| **Speed** | 60% faster training/inference than BERT-base |
| **Performance** | Retains ~97% of BERT-base accuracy on downstream tasks |
| **Cased variant** | Preserves capitalization — important for genre cues (e.g., proper nouns in History/Biography, stylistic casing in Poetry) |
| **Ecosystem** | First-class support in HuggingFace Transformers, well-documented for fine-tuning |

---

## Dataset

**Source:** [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) — Goodreads reviews across 8 genres.

| Genre | Train | Test |
|-------|-------|------|
| poetry | 800 | 200 |
| children | 800 | 200 |
| comics_graphic | 800 | 200 |
| crime | 800 | 200 |
| fantasy_paranormal | 800 | 200 |
| history_biography | 800 | 200 |
| romance | 800 | 200 |
| young_adult | 800 | 200 |
| **Total** | **6,400** | **1,600** |

Data is streamed from UCSD URLs as `.json.gz`, first 10,000 reviews per genre are read, 2,000 randomly sampled, then 1,000 used (800 train / 200 test).

---

## Training Summary

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size (train) | 10 |
| Batch size (eval) | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Optimizer | AdamW |
| Eval strategy | Every 50 steps |
| Trained on | University GPU (SLURM cluster) |
| Training time | ~2-3 min on GPU |

---

## Evaluation Results

### Local Model (Post-Training — Task 6)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.6006** |
| **F1 (macro)** | **0.6023** |
| **Eval Loss** | **1.2716** |
| Runtime | 34.28s (GPU) |

### Hub Model (Re-evaluated — Task 8)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.6006** |
| **F1 (macro)** | **0.6023** |
| **Eval Loss** | **1.2716** |
| Runtime | 229.73s (CPU/Docker) |

### Comparison

Metrics are **identical** between local and Hub evaluation, confirming that model weights were pushed and loaded correctly from Hugging Face. The only difference is inference speed (GPU vs CPU).

### Per-Class Performance

| Genre | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| poetry | 0.83 | 0.80 | 0.81 |
| comics_graphic | 0.83 | 0.81 | 0.82 |
| children | 0.68 | 0.66 | 0.67 |
| history_biography | 0.64 | 0.56 | 0.60 |
| romance | 0.57 | 0.60 | 0.59 |
| crime | 0.57 | 0.60 | 0.58 |
| young_adult | 0.36 | 0.46 | 0.40 |
| fantasy_paranormal | 0.38 | 0.32 | 0.35 |

**Best performing:** Poetry (0.81 F1) and Comics/Graphic (0.82 F1) — genres with distinctive vocabulary.
**Most confused:** Young Adult and Fantasy/Paranormal — genres with overlapping themes and writing styles.

---

## How to Run

### 1. Local Setup

```bash
git clone https://github.com/priyadip/dlops-assignment3.git
cd dlops-assignment3
python -m venv venv && source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Training (University GPU — SLURM)

```bash
sbatch slurm/train.slurm
```

### 3. Local Evaluation 

```bash
python scripts/evaluate.py --model_path ./saved_model --source local
```

### 4. Push to Hugging Face 

```bash
huggingface-cli login
python scripts/push_to_hub.py --repo_id priyadip/goodreads-genre-classifier
```

### 5. Re-evaluate from Hub 

```bash
python scripts/eval_from_hub.py --repo_id priyadip/goodreads-genre-classifier
```

---

## Docker Instructions

### Development / Training Image 

```bash
docker build -t goodreads-train .
docker run --gpus all -it goodreads-train bash

# Inside container:
python scripts/train.py
python scripts/evaluate.py
```

### Production Eval-Only Image 

```bash
# Build
docker build -f Dockerfile.eval \
    --build-arg HF_REPO_ID=priyadip/goodreads-genre-classifier \
    -t goodreads-eval .

# Run — auto-pulls model from HF Hub and evaluates
docker run --rm goodreads-eval

# Run with results saved to host
docker run --rm -v "%cd%\results:/app/results" goodreads-eval
```



## Challenges & Notes

1. **Data streaming:** Reviews are hosted as `.json.gz` files (~GBs each). We stream and sample on-the-fly to avoid downloading entire datasets - only first 10K reviews per genre are read.
2. **Reproducibility:** All random operations are seeded (`SEED=42`). Label maps are sorted alphabetically to ensure consistent encoding across runs and environments.
3. **Model integrity:** Local and Hub evaluations produce identical metrics (accuracy=0.6006, F1=0.6023, loss=1.2716), confirming correct serialization and deserialization.
4. **Genre overlap:** Fantasy/Paranormal and Young Adult are the hardest classes (F1 ~0.35-0.40) due to thematic overlap. More training data or a larger model could improve these.
