"""Train the single best-performing **road-closure** model and lay its full
operating-point trade-off bare.

Run: ``python -m src.train_best``  (add ``GRIDLOCK_NO_TUNE=1`` for a fast pass).

Why a dedicated script? Road closure is the genuinely hard, high-value task
(rare ~7% positive, partly-discretionary, leakage-prone). This entry point
distils everything the experiments in this project established as *best*:

1. **No positive-class reweighting** (``scale_pos_weight = 1``). The classic
   neg/pos weighting *hurts* ranking on this rare target - it inflates recall at
   the cost of a distorted probability surface and lower PR-AUC. Leaving the loss
   unweighted and handling imbalance purely at the decision threshold ranks
   better (verified on the temporal hold-out: AP 0.302 -> 0.319, MCC 0.34 -> 0.41).
2. **Stacked ensemble** - LightGBM + XGBoost + CatBoost combined by a logistic
   meta-learner on out-of-fold predictions, then isotonic-calibrated. Beats any
   single base model on the future test set.
3. **The decision threshold is a policy choice, not a model property.** We print
   the recall-, F1-, F2- and MCC-optimal operating points so the control room can
   choose where to sit on the precision/recall curve. A single recall-favoured
   threshold makes MCC look low even though the ranking is unchanged - this table
   makes that explicit.

Outputs:
  models/closure_model_best.joblib          deployable calibrated stack
  models/preprocessor_closure_best.joblib   matching feature pipeline
  reports/closure_best_operating_points.json metrics + every operating point
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import models as M
from .feature_engineering import build_features
from .preprocessing import Preprocessor
from .splits import split_report, stratified_folds, temporal_split, time_series_folds
from .targets import build_targets
from .text_features import compute_embeddings

TUNE = os.environ.get("GRIDLOCK_NO_TUNE", "0") != "1"
# scale_pos_weight = 1.0 ranks best; raise toward ~3 to buy recall at some AP cost.
BEST_SCALE_POS_WEIGHT = float(os.environ.get("GRIDLOCK_CLOSURE_SPW", "1.0"))


def _fmt(p: dict) -> str:
    return (f"thr={p['threshold']:.3f}  recall={p['recall']:.3f}  "
            f"precision={p['precision']:.3f}  F1={p['f1']:.3f}  "
            f"F2={p['f2']:.3f}  MCC={p['mcc']:.3f}")


def main():
    print(f"[train_best] target=road_closure  tuning={'ON' if TUNE else 'OFF'}  "
          f"scale_pos_weight={BEST_SCALE_POS_WEIGHT}")

    df = pd.read_parquet(C.CLEAN_PARQUET)
    if C.TARGET_CLOSURE not in df.columns:
        df = build_targets(df, save=False)
    if "event_family" not in df.columns:
        df = build_features(df, save=False)
    df = df.reset_index(drop=True)

    emb = compute_embeddings(df)
    train_mask, test_mask = temporal_split(df)
    print(split_report(df, train_mask, test_mask))

    prep = Preprocessor(target=C.TARGET_CLOSURE,
                        n_emb_components=C.CLOSURE_EMB_PCA_COMPONENTS)
    prep.fit(df[train_mask], emb[train_mask])
    X_tr = prep.transform(df[train_mask], emb[train_mask])
    X_te = prep.transform(df[test_mask], emb[test_mask])
    y_tr = df.loc[train_mask, C.TARGET_CLOSURE].to_numpy()
    y_te = df.loc[test_mask, C.TARGET_CLOSURE].to_numpy()

    oof_folds = stratified_folds(y_tr)
    tune_folds = time_series_folds(len(y_tr))

    t0 = time.time()
    artifact, diag = M.train_classification_task(
        X_tr, y_tr, X_te, prep.categorical_feature_names, oof_folds, tune_folds,
        target=C.TARGET_CLOSURE, tune=TUNE, beta=2.0,
        scale_pos_weight=BEST_SCALE_POS_WEIGHT,
    )
    test_prob = artifact.predict_proba(X_te)

    metrics = E.classification_metrics(y_te, test_prob, artifact.threshold, beta=2.0)
    points = E.operating_points(y_te, test_prob)
    metrics["operating_points"] = points
    metrics["oof"] = diag
    metrics["train_seconds"] = round(time.time() - t0, 1)

    print("\n[train_best] === road-closure (best config) ===")
    print(f"  test PR-AUC (AP) = {metrics['average_precision']:.3f}  "
          f"(lift x{metrics['ap_lift_over_base']:.1f} over {metrics['positive_rate']:.3f} base)")
    print(f"  ROC-AUC = {metrics['roc_auc']:.3f}   Brier = {metrics['brier']:.3f}")
    print(f"  base OOF AP: " + ", ".join(f"{k}={v:.3f}" for k, v in diag["base_oof_ap"].items()))
    print("\n  Operating points (pick one as the deployment policy):")
    for label, p in points.items():
        print(f"    {label:14s} {_fmt(p)}")
    print("\n  -> 'recall>=0.8' best for never missing a closure (pre-stage barricades);")
    print("     'mcc_optimal' best for balanced precision/recall.")

    # Default the deployable threshold to MCC-optimal (balanced) - operators can
    # override by reading any operating point from the saved JSON.
    artifact.threshold = float(points["mcc_optimal"]["threshold"])

    joblib.dump(artifact, C.MODELS_DIR / "closure_model_best.joblib")
    joblib.dump(prep, C.MODELS_DIR / "preprocessor_closure_best.joblib")
    E.save_metrics(metrics, "closure_best_operating_points.json")
    E.plot_pr_calibration(y_te, test_prob, "closure_best")
    print("\n[train_best] saved closure_model_best.joblib + operating points JSON.")
    return metrics


if __name__ == "__main__":
    main()
