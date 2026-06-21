"""End-to-end training orchestrator.

Run with ``python -m src.train``. Optional flags via environment variables:
  GRIDLOCK_NO_TUNE=1        skip Optuna (fast smoke run with default params)
  GRIDLOCK_NO_TRANSFORMER=1 use TF-IDF text features instead of the transformer

Produces, under models/ and reports/:
  closure_model.joblib, priority_model.joblib, duration_model.joblib,
  preprocessor_full.joblib, preprocessor_priority.joblib,
  metrics.json, plus PR/calibration/SHAP figures in reports/figures/.
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import models as M
from .feature_engineering import build_features, save_history
from .preprocessing import Preprocessor
from .splits import stratified_folds, temporal_split, time_series_folds, split_report
from .targets import build_targets, winsorized_log_duration
from .text_features import compute_embeddings

TUNE = os.environ.get("GRIDLOCK_NO_TUNE", "0") != "1"


def _load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(C.CLEAN_PARQUET)
    if C.TARGET_CLOSURE not in df.columns:
        df = build_targets(df, save=False)
    if "event_family" not in df.columns:
        df = build_features(df, save=False)
    return df.reset_index(drop=True)


def _train_classification(df, emb, train_mask, test_mask, target, beta):
    # Closure benefits from a wider embedding projection (more rows to fit it);
    # priority is saturated and keeps the default narrow PCA.
    n_emb = C.CLOSURE_EMB_PCA_COMPONENTS if target == C.TARGET_CLOSURE else None
    prep = Preprocessor(target=target, n_emb_components=n_emb)
    X_tr_raw, X_te_raw = df[train_mask], df[test_mask]
    emb_tr, emb_te = emb[train_mask], emb[test_mask]

    prep.fit(X_tr_raw, emb_tr)
    X_tr = prep.transform(X_tr_raw, emb_tr)
    X_te = prep.transform(X_te_raw, emb_te)
    y_tr = df.loc[train_mask, target].to_numpy()
    y_te = df.loc[test_mask, target].to_numpy()

    oof_folds = stratified_folds(y_tr)
    tune_folds = time_series_folds(len(y_tr))
    cat_features = prep.categorical_feature_names

    t0 = time.time()
    artifact, diag = M.train_classification_task(
        X_tr, y_tr, X_te, cat_features, oof_folds, tune_folds,
        target=target, tune=TUNE, beta=beta,
    )
    test_prob = artifact.predict_proba(X_te)
    metrics = E.classification_metrics(y_te, test_prob, artifact.threshold, beta=beta)
    metrics["operating_points"] = E.operating_points(y_te, test_prob)
    metrics["oof"] = diag
    metrics["train_seconds"] = round(time.time() - t0, 1)

    name = "closure" if target == C.TARGET_CLOSURE else "priority"
    E.plot_pr_calibration(y_te, test_prob, name)
    E.plot_shap_summary(artifact.base_models["lightgbm"], X_te, name)
    return artifact, prep, metrics


def _train_duration(df, emb, train_mask, test_mask):
    valid = df["duration_valid"].to_numpy() if "duration_valid" in df else df[C.TARGET_DURATION].notna().to_numpy()
    tr = train_mask & valid
    te = test_mask & valid

    prep = Preprocessor(target=C.TARGET_DURATION)
    prep.fit(df[train_mask], emb[train_mask])  # fit on all train rows for stability
    X_tr = prep.transform(df[tr], emb[tr])
    X_te = prep.transform(df[te], emb[te])

    y_tr_min = df.loc[tr, C.TARGET_DURATION].to_numpy()
    y_te_min = df.loc[te, C.TARGET_DURATION].to_numpy()
    _, cap_minutes = winsorized_log_duration(pd.Series(y_tr_min))  # cap from TRAIN only

    tune_folds = time_series_folds(len(y_tr_min))
    t0 = time.time()
    artifact, diag = M.train_duration_task(X_tr, y_tr_min, cap_minutes, tune_folds, tune=TUNE)

    point = artifact.predict_minutes(X_te)
    quant = artifact.predict_quantiles(X_te)
    metrics = E.regression_metrics(y_te_min, point, quantile_preds=quant)
    metrics["lgb_params"] = diag["lgb_params"]
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["n_train"] = int(tr.sum())
    return artifact, prep, metrics


def main():
    print(f"[train] tuning={'ON' if TUNE else 'OFF'}  transformer={'ON' if C.USE_TRANSFORMER else 'OFF'}")
    df = _load_dataset()
    emb = compute_embeddings(df)
    train_mask, test_mask = temporal_split(df)
    print(split_report(df, train_mask, test_mask))

    all_metrics = {}

    print("\n[train] === road-closure classification ===")
    closure_art, closure_prep, closure_m = _train_classification(
        df, emb, train_mask, test_mask, C.TARGET_CLOSURE, beta=2.0)
    all_metrics["closure"] = closure_m
    print(f"  AP={closure_m['average_precision']:.3f} (lift x{closure_m['ap_lift_over_base']:.1f}) "
          f"recall={closure_m['recall']:.3f} precision={closure_m['precision']:.3f} MCC={closure_m['mcc']:.3f}")

    print("\n[train] === priority classification ===")
    priority_art, priority_prep, priority_m = _train_classification(
        df, emb, train_mask, test_mask, C.TARGET_PRIORITY, beta=1.0)
    all_metrics["priority"] = priority_m
    print(f"  AP={priority_m['average_precision']:.3f} F1={priority_m['f1']:.3f} "
          f"bal_acc={priority_m['balanced_accuracy']:.3f} MCC={priority_m['mcc']:.3f}")

    print("\n[train] === duration regression ===")
    duration_art, duration_prep, duration_m = _train_duration(df, emb, train_mask, test_mask)
    all_metrics["duration"] = duration_m
    print(f"  n_test={duration_m['n']} MAE={duration_m['mae_min']:.0f}min "
          f"median_AE={duration_m['median_ae_min']:.0f}min "
          f"R2_log={duration_m.get('r2_log', float('nan')):.3f} "
          f"coverage80={duration_m.get('interval_coverage_80', float('nan')):.2f}")

    # Persist artifacts.
    joblib.dump(closure_art, C.MODELS_DIR / "closure_model.joblib")
    joblib.dump(priority_art, C.MODELS_DIR / "priority_model.joblib")
    joblib.dump(duration_art, C.MODELS_DIR / "duration_model.joblib")
    joblib.dump(closure_prep, C.MODELS_DIR / "preprocessor_full.joblib")
    joblib.dump(priority_prep, C.MODELS_DIR / "preprocessor_priority.joblib")
    joblib.dump(duration_prep, C.MODELS_DIR / "preprocessor_duration.joblib")
    # Persist the labeled history so inference reproduces the same past-only
    # causal target-rate encodings the models trained on.
    save_history(df)
    E.save_metrics(all_metrics, "metrics.json")
    print("\n[train] artifacts + metrics saved.")
    return all_metrics


if __name__ == "__main__":
    main()
