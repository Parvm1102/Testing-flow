---
title: Gridlock Traffic Intelligence
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
---

<!-- The block above is read by Hugging Face Spaces (Docker SDK). It builds the
     root Dockerfile and routes traffic to app_port (8000). Everything below is
     rendered as the Space description. -->

# Gridlock — Event-Driven Congestion Forecasting & Resource Recommendation


Forecasting the traffic impact of planned and unplanned road events in Bengaluru
(from the anonymised **Astram** event log) and turning those forecasts into
concrete operational recommendations: **manpower, barricading and diversion**.

The dataset ships with no ready-made "impact" label, so the core of this project
is (a) engineering defensible targets, (b) ruthless leakage control, and
(c) skew-aware modelling and evaluation.

---

## 1. Problem framing

> *How can historical and real-time data be used to forecast event-related
> traffic impact and recommend optimal manpower, barricading and diversion plans?*

"Impact" is decomposed into three forecastable targets, each computable **only
from information available when an event is first reported**:

| Task | Target | Type | Drives |
|------|--------|------|--------|
| **T1 Road closure** | `y_road_closure` — will the event need a closure / diversion? | binary (≈7% positive) | barricading + diversion |
| **T2 Priority** | `y_high_priority` — High vs Low operational priority | binary (≈62% positive) | manpower tier |
| **T3 Duration** | `y_duration_min` — how long until cleared (minutes) | regression (heavy-tailed) | manpower + interval |

Three **separately tuned models over one shared, leakage-safe feature pipeline**
(multi-task by construction, not a single fragile multi-head net).

---

## 2. Why this is not a trivial classification task

* **Severe class skew** — closures are ~7% of events, so accuracy is useless. The
  whole evaluation is skew-aware (PR-AUC / average precision, F-beta, MCC,
  balanced accuracy, Brier calibration).
* **No target column** — duration is reconstructed by coalescing
  `resolved → closed → end` timestamps minus start, then cleaned of non-positive
  and automated-batch-closure rows.
* **Leakage everywhere** — many columns are only filled in *after* the event is
  resolved. The single biggest trap: the **end-point coordinates and
  `route_path`** are populated only when a closure/diversion segment is drawn, so
  `has_end_point` alone "predicts" closure at AP ≈ 0.98. These are treated as
  leakage and excluded (see below).
* **Multilingual free text** — descriptions mix English, transliterated Kannada
  and native Kannada script, often stating the impact directly ("road closed",
  "slow moment", "traffic normal"). Encoded with a multilingual
  sentence-transformer plus an interpretable bilingual lexicon.
* **Bimodal duration** — minor incidents clear in minutes–hours; construction
  runs for days–weeks. Handled with a log target, winsorised point model and
  uncapped quantile models for honest prediction intervals.

---

## 3. Leakage control (the most important part)

Columns are partitioned in [src/config.py](src/config.py):

* `ID_COLUMNS` — opaque identifiers, dropped.
* `LEAKAGE_COLUMNS` — known only after the event unfolds; **never** features.
  Includes `status`, all resolution timestamps, `resolved_at_*`, `comment`, and
  critically `endlatitude` / `endlongitude` / `end_address` / `route_path`
  (the closure/diversion geometry — a *consequence* of the decision we predict).
* History features (`hist_hotspot_count`, `loc_event_density`,
  `same_loc_cause_hist`, `same_day_loc_reports`) are computed **strictly
  causally** — each row only sees earlier-reported events.
* All fitted transforms (categorical vocabularies, embedding PCA, numeric
  medians, calibration, decision thresholds) are learned on **training rows
  only**; the chronological test set is never touched until final scoring.
* The literal `corridor` column is excluded from the **priority** model only
  (`PRIORITY_EXCLUDE_FEATURES`) because it makes that label a trivial 1-field
  lookup.

> **Note on priority (T2):** even without `corridor`, priority is ~deterministic
> from *location* (it essentially encodes "is this on a designated priority
> corridor?"), which is legitimately knowable at report time. So its high score
> reflects an genuinely easy geofencing task, **not** leakage — verified via
> feature-importance and per-junction purity checks. The hard ML problems are
> closure (T1) and duration (T3).

---

## 4. Pipeline

```
raw CSV
  └─ data_loading.py     read as strings, strip whitespace
  └─ cleaning.py         parse datetimes, fix coords, flag auto-batch closures
  └─ targets.py          build y_road_closure / y_high_priority / y_duration_min
  └─ feature_engineering temporal + cyclical + spatial(geo_cluster, causal hotspot)
  │                      + recurrence + causal target-rate + bilingual lexicon + missingness
  └─ text_features.py    multilingual sentence-transformer embeddings (cached)
  └─ preprocessing.py    train-fitted assembly: native categs + per-task embedding PCA
  └─ splits.py           chronological train/test + time-series & stratified folds
  └─ models.py           Optuna-tuned LightGBM + XGBoost + CatBoost
  │                      → OOF logistic stack → isotonic calibration → F-beta threshold
  │                      duration: log point model + p10/p50/p90 quantile intervals
  └─ evaluate.py         skew-aware metrics + operating points + PR/calibration/SHAP
  └─ train.py / train_best.py   full 3-task run / focused best closure model
  └─ predict.py / recommend.py   inference → manpower / barricading / diversion
```

### Modelling highlights
* **Stacked ensemble** of three decorrelated gradient-boosters combined by a
  logistic meta-learner trained on **out-of-fold** predictions.
* **Imbalance handled at the threshold, not the loss.** Counter-intuitively, the
  textbook `scale_pos_weight = neg/pos` (≈12.4 here) *hurt* ranking — it inflates
  recall but distorts the probability surface, dropping PR-AUC. We instead leave
  the loss unweighted (`scale_pos_weight = 1`) and absorb the skew purely in the
  **decision threshold**, which lifted test PR-AUC 0.302 → 0.317. The threshold
  itself is a **policy choice** — [src/train_best.py](src/train_best.py) reports
  the recall-, F1-, F2- and MCC-optimal operating points so the control room
  picks where to sit on the curve (e.g. max-recall to never miss a closure vs.
  MCC-optimal for balance).
* **Task-specific embedding width.** The 384-dim multilingual embeddings are the
  single strongest signal, so the PCA width is tuned *per task*. Closure trains
  on all rows and keeps **96** components (validated PR-AUC 0.31 → 0.33); duration
  has only ~2.5k labelled rows, where wide projections overfit, so it keeps a
  narrow **32** (`CLOSURE_EMB_PCA_COMPONENTS` vs `EMBED_PCA_COMPONENTS`). This
  alone is the largest single lever on closure ranking.
* **Causal target-rate features.** Past-only, empirical-Bayes-smoothed closure
  rates per cause / corridor / police-station / geo-cluster / zone / pincode
  (`shift(1)` so a row never sees its own label) give the rare closure target far
  more signal than the static category, and a rolling "ambient duration level"
  tracks the heavy non-stationarity in clearance times. The accumulated history
  is persisted (`history.parquet`) so inference reproduces the exact training-time
  encodings. Together with the wider embedding the deployed closure model reaches
  **PR-AUC 0.326** and duration **log-R² 0.251**.
* **Probability calibration** (isotonic) so the recommendation thresholds act on
  trustworthy probabilities.
* **Uncertainty for duration**: quantile models give an 80% prediction interval,
  not just a point estimate.

---

## 5. Results (chronological hold-out)

Train: 2023-11-09 → 2024-03-14 (n≈6446) · Test: 2024-03-14 → 2024-04-08 (n≈1611).
Full metrics in [reports/metrics.json](reports/metrics.json); figures in
[reports/figures](reports/figures).

**T1 — Road closure** (test positive rate 7.2%; accuracy is meaningless here):

| Metric | Value | Read as |
|--------|-------|---------|
| Average precision (PR-AUC) | **0.326** | **4.5× the 7.2% base rate** |
| ROC-AUC | 0.835 | ranks closures well above non-closures |
| Recall | 0.655 | catches ~2/3 of real closures… |
| Precision | 0.277 | …at the deliberately recall-favoured F2 threshold (0.10) |
| F2 / MCC | 0.515 / 0.360 | recall-weighted; see operating points below |
| Brier | 0.057 | calibrated probabilities |

Confusion @ deployed threshold: TP 76, FP 198, FN 40, TN 1297 — i.e. of 116 real
closures, 76 are pre-flagged for barricading while only 40 are missed.
Base learners (OOF AP): LightGBM 0.380, XGBoost 0.392, CatBoost 0.389 — the
stack combines them to OOF AP 0.396.

> **The threshold is a policy choice, not a fixed model property.** A single
> recall-favoured threshold makes MCC look low even though the *ranking* (PR-AUC)
> is unchanged. The deployed model's full trade-off (in
> [reports/metrics.json](reports/metrics.json); [src/train_best.py](src/train_best.py)
> prints the same for a focused re-tune):
>
> | Operating point | Threshold | Recall | Precision | F1 | F2 | MCC |
> |-----------------|-----------|--------|-----------|----|----|-----|
> | **MCC-optimal** (balanced) | 0.213 | 0.509 | 0.399 | 0.447 | 0.482 | **0.402** |
> | **F2-optimal** (recall-leaning) | 0.097 | 0.655 | 0.277 | 0.390 | **0.515** | 0.360 |
> | **Recall ≥ 0.8** (never miss) | 0.057 | 0.828 | 0.151 | 0.255 | 0.436 | 0.247 |
>
> So the *same* model scores MCC ≈ 0.40 at its balanced operating point — the
> headline 0.36 is simply measured where recall is deliberately favoured.

**T2 — Priority** (genuinely easy report-time geofencing, see note above):

| Metric | Value |
|--------|-------|
| Average precision | **0.984** |
| F1 / balanced-acc | 0.963 / 0.946 |
| MCC / ROC-AUC | 0.901 / 0.981 |
| Brier | 0.039 |

**T3 — Duration** (heavy-tailed minutes; 630 valid test events):

| Metric | Value | Note |
|--------|-------|------|
| Median abs. error | **74 min** | typical incident error |
| MAE / RMSE | 2411 / 5581 min | inflated by multi-week construction tail |
| MAE (log scale) | 1.44 | the honest central-tendency error |
| **R² (log scale)** | **0.251** | honest fit on the heavy-tailed target (raw-minute R² 0.09 is dominated by 140-day outliers) |
| 80% interval coverage | **0.78** | conformalized → ≈ nominal 80% |
| Median interval width | 119 min | actionable uncertainty band |

The closure model's value is best read as **lift over the 7% base rate** at high
recall — exactly what a control room needs to pre-stage barricades. The duration
model's median 73-min error and calibrated 80% interval give a usable clearance
estimate, while the (expected) large MAE honestly reflects the days-long
construction tail.

---

## 6. Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build everything and train (Optuna-tuned). Artifacts land in models/, reports/.
python -m src.train

# Fast smoke run (skip tuning) / offline text features:
GRIDLOCK_NO_TUNE=1 python -m src.train
GRIDLOCK_NO_TRANSFORMER=1 python -m src.train

# Focused best closure model + full operating-point trade-off table:
python -m src.train_best                 # tuned; saves closure_model_best.joblib
GRIDLOCK_CLOSURE_SPW=3 python -m src.train_best   # trade some AP for more recall

# Inference + operational recommendations on raw event rows:
python -m src.predict
```

```python
from src.predict import predict_events
from src.data_loading import load_raw

recs = predict_events(load_raw().tail(20))
print(recs[["closure_probability", "manpower_tier", "officers_suggested",
            "barricading", "diversion", "expected_duration_min"]])
```

---

## 7. Repository layout

```
src/        config, loading, cleaning, targets, features, text, splits,
            preprocessing, models, evaluate, predict, recommend, train
data/raw/   astram_events.csv  (copy of the provided file)
data/processed/  cleaned + feature parquets, cached embeddings
models/     trained artifacts (*.joblib)
reports/    metrics.json + figures/ (PR, calibration, SHAP)
```

---

## 8. Possible extensions
External holiday/festival calendar and weather joins; survival analysis for
still-active (right-censored) events; an online-updating hotspot feed; a serving
API / control-room dashboard.
