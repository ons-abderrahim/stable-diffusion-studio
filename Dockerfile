# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build base image with CUDA + Python
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Optional xFormers for memory-efficient attention
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121 || \
    echo "xFormers install failed — skipping."

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Application
# ─────────────────────────────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

COPY . .

RUN pip install -e . --no-deps

# Create directories
RUN mkdir -p output checkpoints data .cache/huggingface

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
