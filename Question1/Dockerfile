FROM python:3.10-slim

WORKDIR /app

# System deps (for sentencepiece build on some arches)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY translate.py .
COPY evaluate_bleu.py .
COPY input.txt .
COPY reference.txt .

# Default: translate, then compute BLEU
CMD ["bash", "-lc", "python translate.py && python evaluate_bleu.py"]
