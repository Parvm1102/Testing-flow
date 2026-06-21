"""Skew-aware evaluation and explainability.

Accuracy is meaningless on a 7% positive class, so classification is judged on
PR-AUC (average precision), F-beta, MCC, balanced accuracy and calibration
(Brier). Duration is judged on MAE/RMSE in the original minute scale plus pinball
loss and interval coverage for the quantile predictions. SHAP summary plots are
saved for the deployable models.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)

from . import config as C


def classification_metrics(y_true, y_prob, threshold, beta=2.0) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    pos_rate = float(y_true.mean())
    out = {
        "n": int(len(y_true)),
        "positive_rate": pos_rate,
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "ap_lift_over_base": float(average_precision_score(y_true, y_prob) / max(pos_rate, 1e-9)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_true.min() != y_true.max() else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f_beta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def operating_points(y_true, y_prob, recall_target=0.8) -> dict:
    """Report several decision thresholds so the precision/recall/MCC trade-off
    is explicit.

    A single F-beta threshold can make MCC look artificially low even when the
    *ranking* (PR-AUC) is good - MCC and recall trade off against each other.
    This returns the MCC-, F1- and F2-optimal thresholds plus the
    highest-precision threshold that still hits ``recall_target``, so the
    operator can choose where to sit on the curve.
    """
    y_true = np.asarray(y_true)
    grid = np.linspace(0.01, 0.95, 400)

    def stats(t):
        y_pred = (y_prob >= t).astype(int)
        return {
            "threshold": float(t),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        }

    rows = [stats(t) for t in grid]
    pts = {
        "mcc_optimal": max(rows, key=lambda d: d["mcc"]),
        "f1_optimal": max(rows, key=lambda d: d["f1"]),
        "f2_optimal": max(rows, key=lambda d: d["f2"]),
    }
    hit = [r for r in rows if r["recall"] >= recall_target]
    if hit:
        pts[f"recall>={recall_target:g}"] = max(hit, key=lambda d: d["precision"])
    return pts


def _pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def regression_metrics(y_true, y_pred, quantile_preds=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    eps = 1e-6
    out = {
        "n": int(len(y_true)),
        "mae_min": float(mean_absolute_error(y_true, y_pred)),
        "rmse_min": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 2 else float("nan"),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, eps, None)))),
        "median_ae_min": float(np.median(np.abs(y_true - y_pred))),
        # Log-scale errors are more meaningful for a heavy-tailed target whose
        # raw-minute R2 is dominated by a handful of multi-week outliers. The
        # log-scale R2 is the honest goodness-of-fit for this skewed target.
        "mae_log": float(mean_absolute_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))),
        "r2_log": (float(r2_score(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))))
                   if len(y_true) > 2 else float("nan")),
    }
    if quantile_preds is not None:
        lo = np.asarray(quantile_preds[0.1])[mask]
        hi = np.asarray(quantile_preds[0.9])[mask]
        med = np.asarray(quantile_preds[0.5])[mask]
        out["pinball_p50"] = _pinball_loss(y_true, med, 0.5)
        out["interval_coverage_80"] = float(np.mean((y_true >= lo) & (y_true <= hi)))
        out["interval_width_med_min"] = float(np.median(hi - lo))
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_pr_calibration(y_true, y_prob, name: str):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[0].plot(rec, prec, label=f"AP={ap:.3f}")
    axes[0].axhline(np.mean(y_true), ls="--", c="grey", label="base rate")
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"{name}: PR curve")
    axes[0].legend()

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, "o-")
    axes[1].plot([0, 1], [0, 1], ls="--", c="grey")
    axes[1].set(xlabel="Predicted", ylabel="Observed", title=f"{name}: calibration")
    fig.tight_layout()
    path = C.FIGURES_DIR / f"{name}_pr_calibration.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_shap_summary(model, X_sample, name: str, max_display=20):
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):  # binary classifier -> take positive class
            sv = sv[1] if len(sv) > 1 else sv[0]
        shap.summary_plot(sv, X_sample, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        path = C.FIGURES_DIR / f"{name}_shap_summary.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - SHAP can be finicky
        print(f"[evaluate] SHAP failed for {name}: {exc}")
        return None


def save_metrics(metrics: dict, filename: str):
    path = C.REPORTS_DIR / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
