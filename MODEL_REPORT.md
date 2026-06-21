# Gridlock — Model Report & Implementation Guide

A complete account of everything built for the Astram event dataset: the
predicted variables, the models behind each, every metric on the chronological
hold-out, which implementation is best for each task, and the exact single
commands to reproduce the best models.

- **Dataset:** `astram_events.csv` — 8,173 raw → 8,057 clean Bengaluru traffic
  events, 9 Nov 2023 → 8 Apr 2024 (150-day span, 46 raw columns).
- **Two independent implementations:**
  1. **Main pipeline** (`src/`, run with `python -m src.train`) — three
     operational targets over one shared, leakage-safe feature pipeline.
  2. **Standalone hotspot model** (`hotspot_model.py`) — a fourth,
     newly-engineered forward-looking target, fully self-contained.
- **Evaluation discipline (all tasks):** chronological train→test split (train on
  the past, test on the future), skew-aware metrics, leakage-controlled features,
  probability calibration, and decision thresholds chosen on held-out data only.

---

## 0. TL;DR — the four tasks at a glance

| # | Predicted variable | Type (positive rate) | Best implementation | Headline metric | Single command |
|---|--------------------|----------------------|---------------------|-----------------|----------------|
| **T1** | `y_road_closure` — will the event need a closure/diversion? | Binary (7.2%) | Main pipeline | **PR-AUC 0.326** (4.5× base), MCC 0.40 @ balanced | `python -m src.train` |
| **T2** | `y_high_priority` — High vs Low operational priority | Binary (62.2%) | Main pipeline | **PR-AUC 0.984**, MCC 0.90 | `python -m src.train` |
| **T3** | `y_duration_min` — minutes until cleared | Regression (heavy-tailed) | Main pipeline | **log-R² 0.251**, median AE 74 min | `python -m src.train` |
| **T4** | `y_hotspot` — will this ~110 m spot reoffend (≥2 events in 14 days)? | Binary (14.1%) | Standalone hotspot model | **PR-AUC 0.441** (2.8× base), recall 0.88 | `python hotspot_model.py train` |

> **Get every best model in one line:**
> ```bash
> python -m src.train && python hotspot_model.py train
> ```
> The first command trains and deploys the best **T1/T2/T3** models; the second
> trains and deploys the best **T4** model. Details in [§6](#6-how-to-get-the-best-models).

---

## 1. The two implementations

### 1A. Main pipeline — `src/` (`python -m src.train`)

Three **separately tuned models over one shared feature pipeline** (multi-task by
construction, not a single fragile multi-head net). Each model is a **stacked
ensemble**:

```
Optuna-tuned LightGBM  ┐
Optuna-tuned XGBoost   ├─►  logistic meta-learner on out-of-fold preds
Optuna-tuned CatBoost  ┘     └─►  isotonic calibration  └─►  decision threshold
```

- **Duration** uses the same ensemble on a `log1p` target for the point estimate,
  plus **p10/p50/p90 quantile models** with a conformal correction for an honest
  80% prediction interval.
- **Shared features:** temporal + cyclical encodings, spatial bins
  (`geo_cluster` KMeans), causal (past-only) hotspot/recurrence counts,
  empirical-Bayes causal target-rate encodings, multilingual sentence-transformer
  embeddings (per-task PCA width), a bilingual impact lexicon, and missingness
  flags.

### 1B. Standalone hotspot model — `hotspot_model.py`

A single self-contained file (no import from `src/`) with its own cleaning,
target, features, training and prediction CLI. One **calibrated LightGBM**
classifier on a strictly causal feature set, with an isotonic calibrator and a
recall-favouring F2 threshold. Artifacts live under `hotspot_artifacts/`.

---

## 2. T1 — Road closure (`y_road_closure`)

**What it predicts:** at report time, will this event require a road closure or
diversion? Drives **barricading and diversion** planning.
**Model:** stacked LightGBM + XGBoost + CatBoost → logistic meta → isotonic
calibration → F2 threshold. Embedding PCA width **96** (closure-specific).
**Test set:** n = 1,611, positive rate **7.2%** (≈116 real closures).

| Metric | Value | Read as |
|--------|-------|---------|
| **Average precision (PR-AUC)** | **0.326** | **4.5× the 7.2% base rate** |
| ROC-AUC | 0.835 | ranks closures well above non-closures |
| Recall | 0.655 | catches ~2/3 of real closures… |
| Precision | 0.277 | …at the recall-favoured deployed threshold (0.10) |
| F2 / F1 | 0.515 / 0.390 | recall-weighted |
| MCC | 0.360 | at deployed threshold (0.40 at balanced point) |
| Balanced accuracy | 0.761 | |
| Brier | 0.057 | well-calibrated probabilities |
| Confusion @ deploy | TP 76 · FP 198 · FN 40 · TN 1297 | 76/116 closures pre-flagged, 40 missed |

**Base learners (OOF AP):** LightGBM 0.380 · XGBoost 0.392 · CatBoost 0.389 →
stack **0.396**.

**The threshold is a policy choice** — the same model, scored at different
operating points (from [reports/metrics.json](reports/metrics.json)):

| Operating point | Threshold | Recall | Precision | F1 | F2 | MCC |
|-----------------|-----------|--------|-----------|----|----|-----|
| **MCC-optimal** (balanced) | 0.213 | 0.509 | 0.399 | 0.447 | 0.482 | **0.402** |
| **F2-optimal** (recall-leaning, deployed) | 0.097 | 0.655 | 0.277 | 0.390 | **0.515** | 0.360 |
| **Recall ≥ 0.8** (never miss) | 0.057 | 0.828 | 0.151 | 0.255 | 0.436 | 0.247 |

**Why it's hard (and honest):** the strongest raw closure predictors (end-point
coordinates / `route_path`) are *consequences* of the closure decision and were
removed as leakage; closure is partly discretionary; and there is real temporal
drift (OOF AP 0.40 → test 0.33). The PR-AUC of 0.326 is genuine signal at 4.5×
the base rate.

---

## 3. T2 — Priority (`y_high_priority`)

**What it predicts:** High vs Low operational priority. Drives the **manpower
tier**.
**Model:** same stacked ensemble; `corridor` deliberately **excluded** (it makes
the label a trivial one-field lookup). Embedding PCA width 32.
**Test set:** n = 1,611, positive rate **62.2%**.

| Metric | Value |
|--------|-------|
| **Average precision (PR-AUC)** | **0.984** |
| ROC-AUC | 0.980 |
| F1 | 0.963 |
| Precision / Recall | 0.951 / 0.975 |
| Balanced accuracy | 0.946 |
| MCC | 0.901 |
| Brier | 0.040 |
| Confusion @ deploy (thr 0.38) | TP 977 · FP 50 · FN 25 · TN 559 |

**Base learners (OOF AP):** LightGBM 0.991 · XGBoost 0.988 · CatBoost 0.986 →
stack **0.989**.

**Note:** even without `corridor`, priority is near-deterministic from *location*
(it encodes "is this on a designated priority corridor?"), which is legitimately
knowable at report time. The high score reflects a **genuinely easy geofencing
task, not leakage** — verified via feature importance and per-junction purity.
This is the "good" task with little headroom left.

---

## 4. T3 — Duration (`y_duration_min`)

**What it predicts:** minutes from start until the event is cleared
(reconstructed by coalescing `resolved → closed → end` minus start, cleaned of
non-positive and automated batch-closure rows). Drives the **manpower estimate +
clearance interval**.
**Model:** stacked ensemble on a `log1p` target (winsorised p99) for the point
estimate, plus p10/p50/p90 quantile models with conformal correction for the
interval. Embedding PCA width 32.
**Test set:** n = 630 valid events (train n = 2,568).

| Metric | Value | Note |
|--------|-------|------|
| **R² (log scale)** | **0.251** | honest fit on the heavy-tailed target |
| Median absolute error | **74 min** | typical incident error |
| MAE (log scale) | 1.44 | central-tendency error |
| MAE / RMSE (raw min) | 2411 / 5581 | inflated by the multi-week construction tail |
| R² (raw min) | 0.092 | misleading — dominated by 140-day outliers |
| MAPE | 17.2 | |
| 80% interval coverage | **0.78** | conformalized → ≈ nominal 80% |
| Median interval width | 119 min | actionable uncertainty band |

**Why two R² numbers:** raw-minute R² (0.09) is dominated by a handful of
days-to-weeks construction events; the **log-scale R² (0.251)** is the honest
measure of central-tendency fit, and the 74-min median error is what a control
room actually experiences on normal incidents.

---

## 5. T4 — Chronic hotspot early warning (`y_hotspot`) — NEW

**What it predicts (engineered from scratch):** at the moment an event is
reported at a location, will that **same ~110 m spot generate ≥ 2 more events
within the next 14 days?** This is a *recurring-hotspot early warning* — instead
of repeatedly firefighting the same junction/pothole/water-logging spot, the
control room gets a flag to send a **root-cause fix** (drainage, resurfacing, a
permanent marshal).
**Model:** single calibrated LightGBM (isotonic + F2 threshold), built entirely
on strictly past-only causal features.
**Test set:** n = 1,383, base rate **15.6%** (overall positive rate 14.1% across
6,914 labelable rows).

| Metric | Value | Read as |
|--------|-------|---------|
| **Average precision (PR-AUC)** | **0.441** | **2.8× the 15.6% base rate** |
| ROC-AUC | 0.790 | |
| Brier | 0.106 | |
| Recall (deployed F2, thr 0.071) | **0.875** | catches 189 of 216 emerging hotspots |
| Precision (deployed) | 0.259 | early-warning favours recall |
| F1 / MCC (deployed) | 0.400 / 0.299 | |
| Confusion @ deploy | TP 189 · FP 541 · FN 27 · TN 626 | only 27 emerging hotspots missed |

Alternative balanced operating point (MCC-optimal, thr 0.138): recall 0.593,
precision 0.414, MCC **0.381**, F2 0.546.

**Leakage safety:** every feature uses only strictly-earlier events; the label
uses only strictly-later events (disjoint time windows); right-censored rows
(without a full 14-day future) are dropped from train/test. **Top features**
(gain): `junc_win30`, `zone_win30`, `ps_win30`, `police_station`, lat/long,
`loc_days_since_first/last`, `area_to_loc_ratio`, `loc_rate_per_day`.

**Face validity:** the highest-risk events are recurring **construction at HAL
Old Airport** (metro work) and **potholes at J.P. Nagar** — exactly the chronic
spots worth a permanent fix. Honest caveat: brand-new locations (no prior
history) almost never turn chronic (6 positives in 383 first-ever sites), and the
model correctly assigns them low risk rather than inventing signal.

---

## 6. How to get the best models

### Best implementation per task

| Task | Best implementation | Command | Why |
|------|---------------------|---------|-----|
| **T1 Road closure** | Main pipeline (`src/`) | `python -m src.train` | Deploys the spw = 1 + PCA-96 config (PR-AUC 0.326), the best validated closure model. |
| **T2 Priority** | Main pipeline (`src/`) | `python -m src.train` | Only and best implementation (PR-AUC 0.984). |
| **T3 Duration** | Main pipeline (`src/`) | `python -m src.train` | Only and best implementation (log-R² 0.251 + calibrated interval). |
| **T4 Chronic hotspot** | Standalone (`hotspot_model.py`) | `python hotspot_model.py train` | Self-contained forward-looking model (PR-AUC 0.441). |

### Single command for the best models

```bash
# from the repo root, with the venv active:
source .venv/bin/activate

# Best T1 + T2 + T3 (Optuna-tuned) AND best T4, one line:
python -m src.train && python hotspot_model.py train
```

- `python -m src.train` → writes the 6 main model artifacts to `models/`,
  metrics to [reports/metrics.json](reports/metrics.json), and figures to
  `reports/figures/`.
- `python hotspot_model.py train` → writes `hotspot_artifacts/hotspot_bundle.joblib`,
  `hotspot_history.parquet`, and `hotspot_metrics.json`.

### Inference (after training)

```bash
# T1/T2/T3 — closure prob, manpower tier, officers, barricading, diversion, duration interval:
python -m src.predict

# T4 — score events and list the top emerging hotspots:
python hotspot_model.py predict
```

### Useful variants

```bash
GRIDLOCK_NO_TUNE=1 python -m src.train            # fast smoke run (skips Optuna)
GRIDLOCK_NO_TRANSFORMER=1 python -m src.train      # offline TF-IDF text fallback
python -m src.train_best                           # focused closure model + operating-point table
GRIDLOCK_CLOSURE_SPW=3 python -m src.train_best     # trade some PR-AUC for more closure recall
```

---

## 7. Leakage control (applies to all tasks)

- **`LEAKAGE_COLUMNS`** — anything known only after the event unfolds (`status`,
  all resolution timestamps, `resolved_at_*`, `comment`) is never a feature. The
  biggest trap: **end-point coordinates / `route_path`** are populated only when a
  closure/diversion segment is drawn, so `has_end_point` alone "predicts" closure
  at AP ≈ 0.98 — these are excluded as leakage.
- **Causal features only** — every history/recurrence/target-rate feature sees
  strictly earlier events (`shift(1)` / append-after accumulators).
- **Train-only fitting** — categorical vocabularies, embedding PCA, calibration
  and thresholds are learned on training rows; the chronological test set is
  untouched until final scoring.
- **`corridor` excluded from T2 only** — it makes priority a trivial lookup.

---

## 8. Artifact map

| Path | Produced by | Contents |
|------|-------------|----------|
| `models/*.joblib` (6) | `python -m src.train` | T1/T2/T3 models, preprocessors, calibrators, thresholds |
| `models/closure_model_best.joblib` | `python -m src.train_best` | focused closure model |
| `models/geo_kmeans.joblib` | `python -m src.train` | spatial bin assignment for inference |
| `data/processed/history.parquet` | `python -m src.train` | causal target-rate history for inference |
| [reports/metrics.json](reports/metrics.json) | `python -m src.train` | full T1/T2/T3 metrics + operating points |
| `reports/figures/*.png` | `python -m src.train` | PR, calibration, SHAP |
| `hotspot_artifacts/hotspot_bundle.joblib` | `python hotspot_model.py train` | T4 model + isotonic + threshold + dtypes |
| `hotspot_artifacts/hotspot_history.parquet` | `python hotspot_model.py train` | event log for causal feature replay |
| `hotspot_artifacts/hotspot_metrics.json` | `python hotspot_model.py train` | full T4 metrics + operating points |

---

## 9. Summary

- **Four predicted variables**, each engineered from a dataset that ships with no
  ready-made label, each forecastable strictly from report-time information.
- **T2 (priority)** is the saturated "easy" task (0.984); **T1 (closure)** and
  **T3 (duration)** are the genuinely hard operational tasks, improved to PR-AUC
  0.326 and log-R² 0.251 through per-task embedding width and causal target-rate
  features; **T4 (chronic hotspot)** is a new forward-looking early-warning model
  at PR-AUC 0.441 with 0.88 recall.
- **One command per implementation** reproduces every best model:
  `python -m src.train && python hotspot_model.py train`.
