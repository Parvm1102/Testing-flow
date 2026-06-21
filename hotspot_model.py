"""Chronic-Hotspot Early-Warning model (standalone).

This is a SELF-CONTAINED experiment, deliberately independent of the ``src/``
pipeline. It defines its own prediction target, its own feature engineering and
its own train/predict CLI, and shares nothing with the three-task model except
the raw CSV on disk.

------------------------------------------------------------------------------
The target (engineered from scratch)
------------------------------------------------------------------------------
The raw log never says "this place is a chronic problem". We define it:

    At the moment an event is reported at a location, will that SAME ~110 m
    spot generate >= 2 MORE events within the next 14 days?

Operationally this is a *recurring-hotspot early warning*: instead of repeatedly
firefighting the same junction/pothole/water-logging spot, the control room gets
a flag to send a root-cause fix (drainage, resurfacing, a permanent marshal).

Why this target is a good fit for a "skewed, low-data" problem:
  * It is genuinely rare -> ~13-14% positive, a real imbalanced problem.
  * It is strictly forward-looking, so it is useful (not a description of now).
  * Most of its signal lives in engineered *causal* features, which is exactly
    what we want to showcase.

------------------------------------------------------------------------------
Leakage discipline (the whole point of "usable in real life")
------------------------------------------------------------------------------
  * Every feature for an event uses ONLY events strictly earlier in time
    (past-only / causal). No feature ever peeks at the future window.
  * The label uses ONLY events strictly later in time (the next 14 days).
    Features and label therefore live in disjoint time windows.
  * Right-censoring is handled: an event is only *labelable* if we have a full
    14 days of observation after it, otherwise its future is unknown and it is
    dropped from train/test (kept only as history for others).
  * Evaluation is a temporal hold-out (train on the past, test on the future),
    which is the only honest estimate of deployed performance.

Usage
-----
    python hotspot_model.py train      # fit, evaluate, save artifacts
    python hotspot_model.py predict    # score the held-out events + show top risks

Artifacts are written under ``hotspot_artifacts/``.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "astram_events.csv"
RAW_CSV_FALLBACK = next(ROOT.glob("Astram event data*.csv"), None)
ART_DIR = ROOT / "hotspot_artifacts"
ART_DIR.mkdir(exist_ok=True)
BUNDLE_PATH = ART_DIR / "hotspot_bundle.joblib"
HISTORY_PATH = ART_DIR / "hotspot_history.parquet"
METRICS_PATH = ART_DIR / "hotspot_metrics.json"

RANDOM_STATE = 42

# --- target definition --------------------------------------------------- #
GRID_DECIMALS = 3          # 3 dp ~= 110 m cell -> the "location" unit
AREA_DECIMALS = 2          # 2 dp ~= 1.1 km cell -> the "neighbourhood" unit
FUTURE_WINDOW_DAYS = 14    # look-ahead horizon
MIN_FUTURE_EVENTS = 2      # >= this many future events => chronic hotspot

# --- temporal split ------------------------------------------------------ #
TEST_FRACTION = 0.20
CALIB_FRACTION = 0.15      # last slice of train, for early-stop + calibration

# --- causal recency windows (days) --------------------------------------- #
RECENCY_WINDOWS = (7, 14, 30, 90)
NO_HISTORY_SENTINEL = 9999.0

# --- cleaning constants (self-contained copy) ---------------------------- #
NULL_TOKENS = {"NULL", "null", "None", "none", "", "nan", "NaN", "[]"}
LAT_RANGE, LON_RANGE = (12.6, 13.4), (77.2, 77.9)
DATETIME_COLUMNS = ["start_datetime", "created_date"]

EVENT_FAMILY_MAP = {
    "public_event": "gathering", "procession": "gathering", "protest": "gathering",
    "vip_movement": "vip", "construction": "construction",
    "vehicle_breakdown": "breakdown", "accident": "accident",
    "tree_fall": "obstruction", "water_logging": "obstruction", "debris": "obstruction",
    "fog / low visibility": "obstruction",
    "pot_holes": "road_condition", "road_conditions": "road_condition",
    "congestion": "congestion", "test_demo": "other", "others": "other",
}

# Bilingual (English + Kannada) lexicon flags that hint at a *recurring* cause.
LEXICON = {
    "lex_recurring": ["daily", "every day", "everyday", "regular", "always",
                       "frequent", "recurring", "often", "ನಿತ್ಯ", "ಪ್ರತಿದಿನ", "ಯಾವಾಗಲೂ"],
    "lex_construction": ["construction", "metro", "digging", "ongoing", "barricade",
                          "work in progress", "ಕಾಮಗಾರಿ", "ಮೆಟ್ರೋ"],
    "lex_water": ["water", "logging", "flood", "rain", "drain", "ನೀರು", "ಮಳೆ"],
    "lex_pothole": ["pothole", "pot hole", "pot-hole", "gravel", "bad road", "ಗುಂಡಿ"],
    "lex_breakdown": ["breakdown", "break down", "stalled", "puncture", "towing", "ಕೆಟ್ಟು"],
}

CATEGORICAL_FEATURES = [
    "event_type", "event_cause", "event_family", "corridor", "zone",
    "police_station", "junction", "direction", "veh_type", "authenticated",
]


# --------------------------------------------------------------------------- #
# 1. Loading + minimal cleaning (independent of src/)
# --------------------------------------------------------------------------- #
def load_and_clean() -> pd.DataFrame:
    """Load the raw CSV and produce a typed, sane, chronologically-ordered frame."""
    path = RAW_CSV if RAW_CSV.exists() else RAW_CSV_FALLBACK
    if path is None or not Path(path).exists():
        raise FileNotFoundError("Could not locate the Astram events CSV.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    df.columns = [c.strip() for c in df.columns]

    # Normalise the many textual NULL spellings to real NaN.
    df = df.map(lambda v: np.nan if isinstance(v, str) and v.strip() in NULL_TOKENS else v)

    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ["latitude", "longitude", "age_of_truck"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "requires_road_closure" in df.columns:
        df["requires_road_closure"] = (
            df["requires_road_closure"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
        )

    # Keep only plausible Bengaluru coordinates and a usable start time.
    df.loc[~df["latitude"].between(*LAT_RANGE), "latitude"] = np.nan
    df.loc[~df["longitude"].between(*LON_RANGE), "longitude"] = np.nan
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    df = df[df["start_datetime"].notna()].copy()
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="first")

    # Report/creation time drives both ordering and the causal split.
    df["order_time"] = df["created_date"].fillna(df["start_datetime"])
    df = df.sort_values("order_time").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 2. Keys + target
# --------------------------------------------------------------------------- #
def _grid_key(df: pd.DataFrame, decimals: int) -> np.ndarray:
    return (df["latitude"].round(decimals).astype(str) + "_"
            + df["longitude"].round(decimals).astype(str)).to_numpy()


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add future-event count, the binary label, and a labelable (uncensored) mask.

    ``future_count`` for an event = number of LATER events at the same ~110 m
    cell within ``FUTURE_WINDOW_DAYS`` days. ``labelable`` is False for events
    whose 14-day window extends past the last observed timestamp (right-censored).
    """
    df = df.copy()
    key = _grid_key(df, GRID_DECIMALS)
    t = df["order_time"].astype("int64").to_numpy() / 1e9  # epoch seconds
    window = FUTURE_WINDOW_DAYS * 86400

    future_count = np.zeros(len(df), dtype=int)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(key):
        groups[k].append(i)
    for idxs in groups.values():
        times = t[idxs]  # ascending (df is time-sorted)
        for a, row in enumerate(idxs):
            seg = times[a + 1:]
            if seg.size:
                future_count[row] = int(np.searchsorted(seg, times[a] + window, side="right"))

    df["future_count"] = future_count
    df["y_hotspot"] = (future_count >= MIN_FUTURE_EVENTS).astype(int)
    max_t = t.max()
    df["labelable"] = (t + window) <= max_t
    return df


# --------------------------------------------------------------------------- #
# 3. Causal feature engineering (the centrepiece)
# --------------------------------------------------------------------------- #
def _causal_recency(key: np.ndarray, t: np.ndarray, windows=RECENCY_WINDOWS) -> dict:
    """Past-only recency stats per key, computed in a single time-ordered pass.

    For every row returns, using only strictly-earlier rows of the same key:
    prior_count, days_since_last, days_since_first, rate_per_day and the count
    within each trailing window (days).
    """
    n = len(key)
    out = {
        "prior_count": np.zeros(n),
        "days_since_last": np.full(n, NO_HISTORY_SENTINEL),
        "days_since_first": np.zeros(n),
        "rate_per_day": np.zeros(n),
    }
    for w in windows:
        out[f"win{w}"] = np.zeros(n)

    hist: dict[str, list[float]] = defaultdict(list)
    for i in range(n):
        lst = hist[key[i]]
        if lst:
            ti = t[i]
            out["prior_count"][i] = len(lst)
            out["days_since_last"][i] = (ti - lst[-1]) / 86400
            tenure = (ti - lst[0]) / 86400
            out["days_since_first"][i] = tenure
            out["rate_per_day"][i] = len(lst) / max(tenure, 1.0)
            for w in windows:
                lo = ti - w * 86400
                out[f"win{w}"][i] = len(lst) - bisect.bisect_right(lst, lo)
        hist[key[i]].append(t[i])  # append AFTER -> stays strictly causal & sorted
    return out


def _causal_prior_sum(key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Sum of ``value`` over strictly-earlier rows sharing the same key."""
    out = np.zeros(len(key))
    acc: dict[str, float] = defaultdict(float)
    for i in range(len(key)):
        out[i] = acc[key[i]]
        acc[key[i]] += value[i]
    return out


def _causal_distinct(key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Number of DISTINCT ``value`` seen at this key among strictly-earlier rows.

    A spot that has already produced several *different* kinds of incident is a
    structurally bad location, not a one-off."""
    out = np.zeros(len(key))
    seen: dict[str, set] = defaultdict(set)
    for i in range(len(key)):
        out[i] = len(seen[key[i]])
        seen[key[i]].add(value[i])
    return out


def _kannada(s: str) -> int:
    return int(any("\u0c80" <= ch <= "\u0cff" for ch in s))


def assemble_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the full causal feature matrix for every row of ``df`` (time-sorted).

    Pure function: no fitted state. LightGBM consumes the categoricals natively,
    so train/predict only need the model + the historical event log to reproduce
    these features for a new event.
    """
    df = df.reset_index(drop=True)
    t = df["order_time"].astype("int64").to_numpy() / 1e9
    feats = pd.DataFrame(index=df.index)

    # ---- location history (~110 m cell) ---------------------------------- #
    k3 = _grid_key(df, GRID_DECIMALS)
    loc = _causal_recency(k3, t)
    feats["loc_prior_count"] = loc["prior_count"]
    feats["loc_days_since_last"] = loc["days_since_last"]
    feats["loc_days_since_first"] = loc["days_since_first"]
    feats["loc_rate_per_day"] = loc["rate_per_day"]
    feats["loc_win7"] = loc["win7"]
    feats["loc_win14"] = loc["win14"]
    feats["loc_win30"] = loc["win30"]
    feats["loc_win90"] = loc["win90"]
    feats["loc_is_new"] = (loc["prior_count"] == 0).astype(int)

    closure = pd.to_numeric(df.get("requires_road_closure"), errors="coerce").fillna(0).to_numpy()
    feats["loc_prior_closures"] = _causal_prior_sum(k3, closure)
    cause = df["event_cause"].fillna("na").astype(str).to_numpy()
    feats["loc_samecause_prior"] = _causal_prior_sum(
        np.char.add(np.char.add(k3.astype(str), "|"), cause), np.ones(len(df))
    )
    feats["loc_distinct_causes"] = _causal_distinct(k3, cause)
    feats["loc_prior_count_log"] = np.log1p(loc["prior_count"])
    # Acceleration: is this spot heating up right now vs its monthly baseline?
    feats["loc_accel"] = loc["win7"] / (loc["win30"] + 1.0)

    # ---- neighbourhood history (~1.1 km cell) ---------------------------- #
    k2 = _grid_key(df, AREA_DECIMALS)
    area = _causal_recency(k2, t, windows=(7, 30))
    feats["area_prior_count"] = area["prior_count"]
    feats["area_win7"] = area["win7"]
    feats["area_win30"] = area["win30"]
    # Is the broader neighbourhood hotter than this exact spot? (spill-over risk)
    feats["area_to_loc_ratio"] = area["win30"] / (loc["win30"] + 1.0)

    # ---- administrative-area recency ------------------------------------ #
    for col, name in [("police_station", "ps"), ("junction", "junc"), ("zone", "zone")]:
        keys = df[col].fillna("na").astype(str).to_numpy()
        feats[f"{name}_win30"] = _causal_recency(keys, t, windows=(30,))["win30"]

    # ---- temporal (report time) ----------------------------------------- #
    ot = df["order_time"].dt
    hour, dow, month = ot.hour, ot.dayofweek, ot.month
    feats["hour"] = hour
    feats["dow"] = dow
    feats["month"] = month
    feats["weekofyear"] = ot.isocalendar().week.astype(int).to_numpy()
    feats["is_weekend"] = (dow >= 5).astype(int)
    feats["is_morning_peak"] = hour.between(8, 11).astype(int)
    feats["is_evening_peak"] = hour.between(17, 21).astype(int)
    feats["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    lead = (df["start_datetime"] - df["order_time"]).dt.total_seconds() / 3600
    feats["lead_time_hours"] = lead.clip(lower=0).fillna(0)
    feats["has_advance_notice"] = (lead.fillna(0) > 1).astype(int)

    # ---- text-derived scalars ------------------------------------------- #
    desc = df.get("description")
    desc = pd.Series([""] * len(df)) if desc is None else desc.fillna("").astype(str)
    low = desc.str.lower()
    feats["desc_len"] = desc.str.len()
    feats["desc_word_count"] = desc.str.split().map(len)
    feats["desc_has_text"] = (feats["desc_len"] > 0).astype(int)
    feats["desc_is_kannada"] = desc.map(_kannada)
    feats["desc_digit_count"] = desc.str.count(r"\d")
    for name, terms in LEXICON.items():
        feats[name] = low.apply(lambda s, tt=terms: int(any(term in s for term in tt)))

    # ---- raw report-time numerics + missingness ------------------------- #
    feats["latitude"] = df["latitude"].to_numpy()
    feats["longitude"] = df["longitude"].to_numpy()
    feats["age_of_truck"] = pd.to_numeric(df.get("age_of_truck"), errors="coerce").fillna(-1)
    feats["veh_type_missing"] = df.get("veh_type").isna().astype(int) if "veh_type" in df else 0
    feats["corridor_missing"] = df.get("corridor").isna().astype(int) if "corridor" in df else 0

    # ---- categoricals (LightGBM-native) --------------------------------- #
    fam = df["event_cause"].astype(str).str.strip().str.lower().map(EVENT_FAMILY_MAP).fillna("other")
    # Causes that are structurally recurring (potholes/water-logging/construction/
    # tree-fall) flag a place far more likely to re-offend than a one-off accident.
    feats["chronic_prone_cause"] = fam.isin(
        ["road_condition", "obstruction", "construction"]
    ).astype(int).to_numpy()
    df = df.assign(event_family=fam)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns or c == "event_family"]
    for c in cat_cols:
        feats[c] = df[c].astype("object").where(df[c].notna(), other=np.nan)

    num_cols = [c for c in feats.columns if c not in cat_cols]
    return feats, num_cols, cat_cols


# --------------------------------------------------------------------------- #
# 4. Categorical encoding (stable across train/predict)
# --------------------------------------------------------------------------- #
def fit_category_dtypes(feats: pd.DataFrame, cat_cols: list[str]) -> dict:
    return {c: pd.CategoricalDtype(categories=sorted(feats[c].dropna().unique())) for c in cat_cols}


def apply_category_dtypes(feats: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    feats = feats.copy()
    for c, dt in dtypes.items():
        feats[c] = feats[c].astype(dt)
    return feats


# --------------------------------------------------------------------------- #
# 5. Evaluation helpers
# --------------------------------------------------------------------------- #
def operating_points(y, p, recall_target=0.7) -> dict:
    """Scan thresholds and return MCC-optimal / F2-optimal / recall>=target points."""
    grid = np.linspace(0.01, 0.95, 200)
    best = {"mcc": (-2, None), "f2": (-1, None), "rec": (None, None)}
    for thr in grid:
        yh = (p >= thr).astype(int)
        if yh.sum() == 0:
            continue
        prec = precision_score(y, yh, zero_division=0)
        rec = recall_score(y, yh, zero_division=0)
        mcc = matthews_corrcoef(y, yh) if len(np.unique(yh)) > 1 else 0.0
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        if mcc > best["mcc"][0]:
            best["mcc"] = (mcc, dict(threshold=float(thr), precision=float(prec),
                                     recall=float(rec), mcc=float(mcc), f2=float(f2)))
        if f2 > best["f2"][0]:
            best["f2"] = (f2, dict(threshold=float(thr), precision=float(prec),
                                   recall=float(rec), mcc=float(mcc), f2=float(f2)))
        if rec >= recall_target and (best["rec"][0] is None or prec > best["rec"][0]):
            best["rec"] = (prec, dict(threshold=float(thr), precision=float(prec),
                                      recall=float(rec), mcc=float(mcc), f2=float(f2)))
    return {"mcc_optimal": best["mcc"][1], "f2_optimal": best["f2"][1],
            f"recall>={recall_target}": best["rec"][1]}


def evaluate(y, p, threshold) -> dict:
    yh = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    base = float(np.mean(y))
    prec = precision_score(y, yh, zero_division=0)
    return {
        "n": int(len(y)), "base_rate": base,
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(prec), "recall": float(recall_score(y, yh, zero_division=0)),
        "f1": float(f1_score(y, yh, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yh)) if len(np.unique(yh)) > 1 else 0.0,
        "precision_lift_over_base": float(prec / base) if base else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------------- #
# 6. Train
# --------------------------------------------------------------------------- #
def train() -> None:
    print("[hotspot] loading + cleaning ...")
    df = load_and_clean()
    df = build_target(df)
    feats_all, num_cols, cat_cols = assemble_features(df)

    # Keep only uncensored (labelable) rows for modelling; the rest still served
    # as history when building the causal features above.
    mask = df["labelable"].to_numpy()
    feats = feats_all[mask].reset_index(drop=True)
    y = df.loc[mask, "y_hotspot"].to_numpy()
    order = df.loc[mask, "order_time"].to_numpy()
    print(f"[hotspot] labelable rows {len(feats)} of {len(df)}  positive rate {y.mean():.3f}")

    # Temporal split: past -> train, future -> test.
    n = len(feats)
    cut = int(n * (1 - TEST_FRACTION))
    Xtr_all, ytr_all = feats.iloc[:cut], y[:cut]
    Xte, yte = feats.iloc[cut:], y[cut:]
    # Carve a time-ordered calibration tail out of train (early-stop + isotonic).
    cal_cut = int(len(Xtr_all) * (1 - CALIB_FRACTION))
    Xtr, ytr = Xtr_all.iloc[:cal_cut], ytr_all[:cal_cut]
    Xcal, ycal = Xtr_all.iloc[cal_cut:], ytr_all[cal_cut:]
    print(f"[hotspot] train {len(Xtr)}  calib {len(Xcal)}  test {len(Xte)}")

    dtypes = fit_category_dtypes(feats, cat_cols)
    Xtr, Xcal, Xte = (apply_category_dtypes(x, dtypes) for x in (Xtr, Xcal, Xte))

    model = LGBMClassifier(
        objective="binary", n_estimators=800, learning_rate=0.02, num_leaves=31,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(
        Xtr, ytr, eval_set=[(Xcal, ycal)], eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
    )

    # Calibrate probabilities on the held-out calibration slice (isotonic).
    raw_cal = model.predict_proba(Xcal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, ycal)
    p_cal = iso.predict(raw_cal)

    # Choose the deployed threshold on calibration data (never on test). For an
    # early-warning use case, missing an emerging hotspot costs more than a
    # wasted inspection, so we deploy the recall-favouring F2-optimal point
    # (also far more stable across the temporal shift than the sharp MCC peak).
    ops_cal = operating_points(ycal, p_cal)
    deployed_threshold = ops_cal["f2_optimal"]["threshold"]

    # Honest test evaluation.
    p_test = iso.predict(model.predict_proba(Xte)[:, 1])
    test_metrics = evaluate(yte, p_test, deployed_threshold)
    test_ops = operating_points(yte, p_test)

    # Cold-start view: rows whose location had NO prior history (the hard,
    # genuinely useful case where pure autocorrelation cannot help).
    cold = Xte["loc_is_new"].to_numpy() == 1
    cold_metrics = None
    if cold.sum() > 30:
        n_pos = int(yte[cold].sum())
        cold_metrics = {
            "n": int(cold.sum()), "n_pos": n_pos, "base_rate": float(yte[cold].mean()),
        }
        if n_pos >= 15 and len(np.unique(yte[cold])) > 1:
            cold_metrics["average_precision"] = float(average_precision_score(yte[cold], p_test[cold]))
            cold_metrics["roc_auc"] = float(roc_auc_score(yte[cold], p_test[cold]))

    importance = (pd.Series(model.feature_importances_, index=Xtr.columns)
                  .sort_values(ascending=False))

    # Persist a compact history so a NEW event can reproduce its causal features.
    hist_cols = ["latitude", "longitude", "order_time", "start_datetime",
                 "event_cause", "requires_road_closure", "police_station",
                 "junction", "zone", "description"]
    df[[c for c in hist_cols if c in df.columns]].to_parquet(HISTORY_PATH, index=False)

    bundle = {
        "model": model, "isotonic": iso, "threshold": deployed_threshold,
        "cat_dtypes": dtypes, "feature_cols": list(feats.columns),
        "num_cols": num_cols, "cat_cols": cat_cols,
        "config": {
            "grid_decimals": GRID_DECIMALS, "area_decimals": AREA_DECIMALS,
            "future_window_days": FUTURE_WINDOW_DAYS, "min_future_events": MIN_FUTURE_EVENTS,
        },
    }
    joblib.dump(bundle, BUNDLE_PATH)

    metrics = {
        "target": f">= {MIN_FUTURE_EVENTS} future events within {FUTURE_WINDOW_DAYS}d "
                  f"at a ~110m cell",
        "n_labelable": int(n), "positive_rate": float(y.mean()),
        "deployed_threshold": deployed_threshold,
        "test": test_metrics, "test_operating_points": test_ops,
        "cold_start_test": cold_metrics,
        "top_features": importance.head(20).round(1).to_dict(),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    _print_report(metrics, importance)


def _print_report(m: dict, importance: pd.Series) -> None:
    t = m["test"]
    print("\n" + "=" * 70)
    print("CHRONIC-HOTSPOT EARLY WARNING  —  temporal hold-out")
    print("=" * 70)
    print(f"target            : {m['target']}")
    print(f"positive rate     : {m['positive_rate']:.3f}   (n_labelable={m['n_labelable']})")
    print(f"PR-AUC (AP)       : {t['average_precision']:.3f}   "
          f"(base {t['base_rate']:.3f} -> lift x{t['average_precision']/t['base_rate']:.1f})")
    print(f"ROC-AUC           : {t['roc_auc']:.3f}")
    print(f"Brier             : {t['brier']:.3f}")
    print(f"deployed thr (F2) : {t['threshold']:.3f}")
    print(f"  precision {t['precision']:.3f}  recall {t['recall']:.3f}  "
          f"F1 {t['f1']:.3f}  MCC {t['mcc']:.3f}  (precision lift x{t['precision_lift_over_base']:.1f})")
    c = t["confusion"]
    print(f"  confusion  TP {c['tp']}  FP {c['fp']}  FN {c['fn']}  TN {c['tn']}")
    print("\noperating points (test):")
    for name, op in m["test_operating_points"].items():
        if op:
            print(f"  {name:14s} thr {op['threshold']:.3f}  recall {op['recall']:.3f}  "
                  f"precision {op['precision']:.3f}  MCC {op['mcc']:.3f}")
    if m["cold_start_test"]:
        cs = m["cold_start_test"]
        line = (f"\ncold-start (first-ever event at the location) test subset: "
                f"n={cs['n']}  positives={cs['n_pos']}  base {cs['base_rate']:.3f}")
        if "average_precision" in cs:
            line += f"  AP {cs['average_precision']:.3f}  ROC {cs['roc_auc']:.3f}"
        else:
            line += "  (too few positives to score — brand-new spots rarely turn chronic)"
        print(line)
    print("\ntop 15 features (gain importance):")
    for name, val in importance.head(15).items():
        print(f"  {name:24s} {val:8.0f}")
    print("=" * 70)
    print(f"artifacts -> {ART_DIR}")


# --------------------------------------------------------------------------- #
# 7. Predict
# --------------------------------------------------------------------------- #
def score_events(new_df: pd.DataFrame) -> pd.DataFrame:
    """Score new events. Causal features are rebuilt from the saved history so a
    fresh event sees the same accumulated past the model trained on."""
    bundle = joblib.load(BUNDLE_PATH)
    history = pd.read_parquet(HISTORY_PATH)

    new = new_df.copy()
    new["__is_new__"] = True
    history["__is_new__"] = False
    combined = pd.concat([history, new], ignore_index=True)
    combined["order_time"] = pd.to_datetime(combined["order_time"], utc=True)
    combined = combined.sort_values("order_time").reset_index(drop=True)

    feats, _, _ = assemble_features(combined)
    feats = apply_category_dtypes(feats, bundle["cat_dtypes"])[bundle["feature_cols"]]
    out_mask = combined["__is_new__"].to_numpy()

    raw = bundle["model"].predict_proba(feats[out_mask])[:, 1]
    prob = bundle["isotonic"].predict(raw)
    res = new_df.copy().reset_index(drop=True)
    res["hotspot_risk"] = prob
    res["hotspot_flag"] = (prob >= bundle["threshold"]).astype(int)
    return res


def predict() -> None:
    """Demo: rebuild the held-out events and score them, then show the top risks."""
    if not BUNDLE_PATH.exists():
        raise SystemExit("No trained model found. Run `python hotspot_model.py train` first.")
    bundle = joblib.load(BUNDLE_PATH)

    df = build_target(load_and_clean())
    feats_all, _, _ = assemble_features(df)
    mask = df["labelable"].to_numpy()
    n = int(mask.sum())
    cut = int(n * (1 - TEST_FRACTION))

    df_lab = df[mask].reset_index(drop=True)
    feats = apply_category_dtypes(
        feats_all[mask].reset_index(drop=True), bundle["cat_dtypes"]
    )[bundle["feature_cols"]]

    te_idx = np.arange(cut, n)
    raw = bundle["model"].predict_proba(feats.iloc[te_idx])[:, 1]
    prob = bundle["isotonic"].predict(raw)
    thr = bundle["threshold"]

    view = df_lab.iloc[te_idx][[
        "order_time", "latitude", "longitude", "event_cause", "police_station",
        "junction", "y_hotspot",
    ]].copy()
    view["risk"] = prob
    view["flag"] = (prob >= thr).astype(int)
    flagged = int(view["flag"].sum())
    print(f"[hotspot] scored {len(view)} held-out events; flagged {flagged} "
          f"as emerging hotspots (thr={thr:.3f})")
    print("\nTop 10 highest-risk locations (forward-looking):")
    cols = ["order_time", "event_cause", "police_station", "junction", "risk", "y_hotspot"]
    top = view.sort_values("risk", ascending=False).head(10)[cols]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(top.to_string(index=False))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Chronic-hotspot early-warning model.")
    ap.add_argument("command", choices=["train", "predict"], help="action to run")
    args = ap.parse_args()
    if args.command == "train":
        train()
    else:
        predict()


if __name__ == "__main__":
    main()
