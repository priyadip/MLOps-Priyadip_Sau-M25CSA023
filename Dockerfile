# ============================================================
# Dockerfile — Development / Training image
# ============================================================
# Build:  docker build -t goodreads-train .
# Run:    docker run --gpus all -it goodreads-train bash
# ============================================================

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python deps first (Docker cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create dirs that the pipeline expects
RUN mkdir -p data results saved_model

# Sanity check
RUN python -c "import torch; import transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"

# Default: drop into bash so user can run any script
CMD ["bash"]
