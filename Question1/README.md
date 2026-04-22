# Question 1 — Bengali → English Translation

## Files
- `translate.py` — runs Helsinki-NLP/opus-mt-bn-en on `input.txt`, writes `output.txt`
- `evaluate_bleu.py` — computes corpus BLEU between `output.txt` and `reference.txt`
- `Dockerfile` — reproducible container
- `requirements.txt` — pip deps

## Steps
1. Download `input.txt` and `reference.txt` from the Google Drive link in the question paper and place them in this folder (overwrite the placeholders).
2. Native run:
   ```bash
   pip install -r requirements.txt
   python translate.py
   python evaluate_bleu.py
   ```
3. Docker run:
   ```bash
   docker build -t q1-translate .
   docker run --rm -v "$PWD":/app q1-translate
   ```
   (The `-v` mount lets Docker read your real `input.txt`/`reference.txt` and write `output.txt` back to your host.)

## Answers to fill in README.md of repo root

- **First statement translation:** (printed at the end of `translate.py`)
- **BLEU score:** (printed at the end of `evaluate_bleu.py`)
