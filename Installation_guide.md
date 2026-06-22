# Installation & Setup Guide

This guide walks you through installing the project, running the web application,
and training the **best road-closure model from scratch**.

- **Repository:** <https://github.com/Parvm1102/Testing-flow>

---

## 1. Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10 – 3.12 | Required for the ML pipeline and backend |
| pip / venv | latest | Bundled with Python |
| Node.js | 18+ | Required only to build/serve the frontend |
| npm | 9+ | Ships with Node.js |
| Git + Git LFS | latest | **Git LFS is required** — model artifacts are stored as LFS objects |
| Docker | optional | For containerized deployment |

> **Important:** The trained model files (`models/*.joblib`,
> `hotspot_artifacts/*`, `data/processed/history.parquet`) are tracked with
> **Git LFS**. A plain clone only fetches small pointer stubs (~132 bytes), which
> will cause inference to fail with a 500 error. Always run `git lfs pull`.

---

## 2. Clone the repository

```bash
git clone https://github.com/Parvm1102/Testing-flow.git
cd Testing-flow

# Fetch the real model artifacts (NOT the LFS pointer stubs)
git lfs install
git lfs pull
```

To confirm the artifacts are real and not stubs, check that the files are large
(MBs, not bytes):

```bash
ls -lh models/*.joblib
```

---

## 3. Python environment & dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

This installs the full stack: pandas/numpy/scikit-learn, the gradient boosters
(LightGBM, XGBoost, CatBoost), Optuna for tuning, and
`sentence-transformers` + `torch` for the multilingual text embeddings.

---

## 4. Run the web application (UI + API)

The backend (FastAPI) serves both the prebuilt frontend and precomputed JSON.

### 4a. Build the frontend

```bash
cd app/frontend
npm install
npm run build          # runs `tsc -b && vite build`, output -> app/frontend/dist
cd ../..
```

### 4b. Start the backend

```bash
source .venv/bin/activate
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000> in your browser.

- The map, top-areas, and metrics views work from precomputed JSON and need **no**
  ML dependencies.
- The `/api/predict` endpoint lazily loads the ML models. If it returns a `500`,
  the most common cause is missing LFS artifacts — re-run `git lfs pull`.

### 4c. (Optional) Run with Docker

The `Dockerfile` lives at the **repository root** and builds the frontend + backend
into a single image.

```bash
git lfs pull                       # ensure real artifacts before building
docker build -t testing-flow:latest .
docker run -p 8000:8000 testing-flow:latest
```

---

## 5. Train the best model from scratch

The "best" model is the focused, Optuna-tuned **road-closure** stacked ensemble
produced by `src/train_best.py`. It outputs:

- `models/closure_model_best.joblib` — the deployable, calibrated stacked ensemble
- `models/preprocessor_closure_best.joblib` — the matching feature pipeline
- `reports/closure_best_operating_points.json` — metrics + every operating point

### Step 0 — Provide the raw data

Ensure the raw event log is present at:

```
data/raw/astram_events.csv
```

(The repository already ships a copy. Replace it to retrain on newer data.)

### Step 1 — Clean the raw data

This parses timestamps, fixes coordinates, flags automated batch closures, and
writes the cleaned parquet (`data/processed/events_clean.parquet`):

```bash
source .venv/bin/activate
python -m src.cleaning
```

### Step 2 — Train the best closure model

```bash
python -m src.train_best
```

What this run does:

1. Builds leakage-safe features (temporal, cyclical, spatial geo-clusters, causal
   hotspot counts, empirical-Bayes target-rate encodings, bilingual lexicon).
2. Computes multilingual sentence-transformer embeddings (cached to
   `data/processed/text_embeddings.npy` on first run).
3. Performs a chronological train/test split.
4. Optuna-tunes and trains a **LightGBM + XGBoost + CatBoost** ensemble combined
   by a logistic meta-learner on out-of-fold predictions, then isotonic-calibrates.
5. Prints and saves the recall-, F1-, F2- and MCC-optimal operating points so you
   can pick a deployment threshold as a policy choice.

### Useful training flags

```bash
# Fast smoke run — skip Optuna tuning (uses sensible default hyperparameters)
GRIDLOCK_NO_TUNE=1 python -m src.train_best

# Trade some PR-AUC for higher recall (raise positive-class weight toward ~3)
GRIDLOCK_CLOSURE_SPW=3 python -m src.train_best
```

### (Optional) Train all three tasks

To retrain the full multi-task suite (closure, priority, duration) and regenerate
`reports/metrics.json` plus PR/calibration/SHAP figures:

```bash
python -m src.train

# Faster variants:
GRIDLOCK_NO_TUNE=1 python -m src.train          # skip Optuna
GRIDLOCK_NO_TRANSFORMER=1 python -m src.train   # TF-IDF instead of transformer
```

---

## 6. Run inference / recommendations

After training (or with the shipped artifacts):

```bash
python -m src.predict
```

Or from Python:

```python
from src.predict import predict_events
from src.data_loading import load_raw

recs = predict_events(load_raw().tail(20))
print(recs[["closure_probability", "manpower_tier", "officers_suggested",
            "barricading", "diversion", "expected_duration_min"]])
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/api/predict` returns 500; `KeyError` on `joblib.load` | LFS pointer stubs instead of real models | `git lfs pull` |
| Frontend shows an old/stale UI in Docker | A stale cached image was reused | Rebuild: `docker build -t testing-flow:latest .` |
| `ModuleNotFoundError` for torch/lightgbm/etc. | Dependencies not installed in the active venv | `source .venv/bin/activate && pip install -r requirements.txt` |
| `FileNotFoundError: events_clean.parquet` during training | Cleaning step skipped | Run `python -m src.cleaning` first |
| First training run is slow | Downloading + computing transformer embeddings | Embeddings are cached afterward in `data/processed/` |
