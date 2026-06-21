# syntax=docker/dockerfile:1
#
# Gridlock Traffic Intelligence — single-image demo.
# One container serves the React frontend AND the FastAPI inference API on one
# port. Runs unchanged on Hugging Face Spaces (Docker SDK), Render, or any host.
#
# This Dockerfile lives at the repo root because Hugging Face Spaces builds the
# `Dockerfile` at the root with the repo as context. Build context = repo root:
#     docker build -t gridlock .
#     docker run -p 8000:8000 gridlock     # -> http://localhost:8000
#
# ---------------------------------------------------------------------------- #
# Stage 1 — build the frontend with Node, output static assets to /build/dist
# ---------------------------------------------------------------------------- #
FROM node:20-slim AS frontend

WORKDIR /build

# Install deps first (cached unless the lockfile changes).
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci

# Build the production bundle.
COPY app/frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------- #
# Stage 2 — Python runtime serving API + the built frontend
# ---------------------------------------------------------------------------- #
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by LightGBM / XGBoost at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch wheel first (avoids pulling ~2GB of CUDA libs),
# then the remaining inference dependencies. torch==2.12.1 in requirements is
# then already satisfied by the +cpu build, so it is not re-resolved from PyPI.
# pip is upgraded first because the 23.0.1 shipped in slim rejects the PyTorch
# index's `typing_extensions` wheels (name-normalization bug). xgboost declares
# nvidia-nccl-cu12 (~300MB) for multi-GPU only; CPU inference never loads it, so
# we drop it (scoped `|| true` keeps the build green if it is already absent).
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && { pip uninstall -y nvidia-nccl-cu12 || true; }

# Hugging Face Spaces runs the container as UID 1000. Create that user and switch
# to it BEFORE downloading the model + copying code, so the HF cache and app
# files are owned by the runtime user without a costly `chown -R` layer. This is
# also valid on Render / any host (uvicorn binds a high port, so no root needed).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

# Pre-download the multilingual sentence-transformer into the image (owned by
# `user`) so the first /api/predict is fast and the container needs no network.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# ---- Application code + the artifacts inference actually loads (owned by user) ---- #
COPY --chown=user:user src/ ./src/
COPY --chown=user:user hotspot_model.py ./hotspot_model.py
COPY --chown=user:user app/__init__.py ./app/__init__.py
COPY --chown=user:user app/backend/ ./app/backend/

# Trained models + preprocessors (T1–T4) and the geo KMeans.
COPY --chown=user:user models/ ./models/

# Hotspot bundle + replay history (the raw training CSV is NOT shipped).
COPY --chown=user:user hotspot_artifacts/hotspot_bundle.joblib hotspot_artifacts/hotspot_history.parquet hotspot_artifacts/hotspot_metrics.json ./hotspot_artifacts/

# Causal target-rate history used by the feature pipeline at inference time.
COPY --chown=user:user data/processed/history.parquet ./data/processed/history.parquet

# Report figures (served by /api/figures) + metric JSON.
COPY --chown=user:user reports/ ./reports/

# Built frontend from stage 1.
COPY --chown=user:user --from=frontend /build/dist ./app/frontend/dist

# Build-time smoke test: load every pickled artifact (T1-T4 models, preprocessors,
# hotspot bundle + histories) through the real inference service. This fails the
# build early if there is any library-version skew or a missing file, rather than
# surfacing it at runtime on the first request.
RUN python -c "from app.backend.inference import get_service; get_service(); print('Inference artifacts load OK')"

# Skip network calls to Hugging Face now that the model is cached in the image.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8000

# HF Spaces routes to the port declared as `app_port` in README.md (8000 here);
# Render injects $PORT. Both are honoured by the CMD below.
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
