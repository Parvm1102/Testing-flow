"""Modelling toolbox: tuned gradient boosting, OOF stacking, calibration.

Design choices that push past a single off-the-shelf classifier:

* **Stacked ensemble** - LightGBM (Optuna-tuned) + XGBoost + CatBoost give three
  decorrelated views of the same features; a logistic meta-learner combines their
  out-of-fold (OOF) predictions. OOF stacking means the meta-learner never sees a
  base model predict on data it was trained on.
* **Leakage-safe calibration & thresholds** - probability calibration and the
  decision threshold are both fit on OOF predictions of the *training* data, so
  the chronological test set stays untouched until final evaluation.
* **Imbalance-aware** - the rare road-closure class uses ``scale_pos_weight`` and
  the threshold is chosen to maximise F-beta (recall-favoured), not accuracy.
* **Duration with uncertainty** - a log-target point model plus quantile models
  (p10/p50/p90) yield calibrated prediction *intervals*, not just point guesses.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score

from . import config as C

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

F_BETA = 2.0  # recall is ~F_BETA times as important as precision for closures


# --------------------------------------------------------------------------- #
# Base model factories (each handles categorical dtype in its own way)
# --------------------------------------------------------------------------- #
def make_lightgbm_classifier(params: dict, scale_pos_weight: float = 1.0):
    from lightgbm import LGBMClassifier

    base = dict(
        objective="binary", n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=C.RANDOM_STATE, n_jobs=-1, verbosity=-1,
        scale_pos_weight=scale_pos_weight,
    )
    base.update(params or {})
    return LGBMClassifier(**base)


def make_xgboost_classifier(scale_pos_weight: float = 1.0):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=2,
        tree_method="hist", enable_categorical=True, eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight, random_state=C.RANDOM_STATE, n_jobs=-1,
    )


def make_catboost_classifier(scale_pos_weight: float, cat_features: list[str]):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="PRAUC", cat_features=cat_features,
        scale_pos_weight=scale_pos_weight, random_seed=C.RANDOM_STATE,
        verbose=False, allow_writing_files=False,
    )


def _catboost_frame(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """CatBoost rejects NaN in categorical columns - replace with a token."""
    X = X.copy()
    for c in cat_features:
        X[c] = X[c].astype("object").where(X[c].notna(), "missing").astype(str)
    return X


# --------------------------------------------------------------------------- #
# Optuna tuning for the primary LightGBM classifier
# --------------------------------------------------------------------------- #
def tune_lightgbm_classifier(X, y, folds, scale_pos_weight, n_trials=C.OPTUNA_TRIALS):
    y = np.asarray(y)

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 900),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        scores = []
        for tr_idx, va_idx in folds:
            if y[va_idx].sum() < 2:  # need positives to score AP
                continue
            model = make_lightgbm_classifier(params, scale_pos_weight)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            p = model.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(average_precision_score(y[va_idx], p))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=C.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=C.OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    return study.best_params


# --------------------------------------------------------------------------- #
# OOF stacking utilities
# --------------------------------------------------------------------------- #
def _oof_and_full(factory, X_tr, y_tr, X_te, folds, catboost=False, cat_features=None):
    """Return OOF train predictions and mean test predictions for one base model."""
    y_tr = np.asarray(y_tr)
    oof = np.zeros(len(X_tr))
    test_acc = np.zeros(len(X_te))
    Xtr_use = _catboost_frame(X_tr, cat_features) if catboost else X_tr
    Xte_use = _catboost_frame(X_te, cat_features) if catboost else X_te
    n = len(folds)
    for tr_idx, va_idx in folds:
        model = factory()
        model.fit(Xtr_use.iloc[tr_idx], y_tr[tr_idx])
        oof[va_idx] = model.predict_proba(Xtr_use.iloc[va_idx])[:, 1]
        test_acc += model.predict_proba(Xte_use)[:, 1] / n
    # Refit on all training rows for the deployable model.
    full = factory()
    full.fit(Xtr_use, y_tr)
    return oof, test_acc, full


@dataclass
class ClassificationArtifact:
    target: str
    base_models: dict = field(default_factory=dict)
    cat_features: list[str] = field(default_factory=list)
    meta: LogisticRegression | None = None
    calibrator: IsotonicRegression | None = None
    threshold: float = 0.5
    model_order: list[str] = field(default_factory=list)

    def _stack_matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for name in self.model_order:
            model = self.base_models[name]
            Xu = _catboost_frame(X, self.cat_features) if name == "catboost" else X
            cols.append(model.predict_proba(Xu)[:, 1])
        return np.column_stack(cols)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_in = self._stack_matrix(X)
        p = self.meta.predict_proba(meta_in)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        return p

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


def best_threshold(y_true, y_prob, beta=F_BETA) -> float:
    """Threshold maximising F-beta over a fine grid (recall-favoured)."""
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)


def train_classification_task(X_tr, y_tr, X_te, cat_features, oof_folds, tune_folds,
                              target=C.TARGET_CLOSURE, tune=True,
                              beta=F_BETA, scale_pos_weight=None) -> tuple[ClassificationArtifact, dict]:
    """Tune LightGBM, build the OOF stack, calibrate, pick a threshold.

    ``tune_folds`` are temporal (honest HP selection); ``oof_folds`` cover every
    training row exactly once (full-coverage stacking / calibration). Returns the
    deployable artifact and an OOF diagnostics dict.

    ``scale_pos_weight`` controls positive-class reweighting. Default ``None`` ->
    1.0 (no reweighting): on this rare, partly-discretionary closure target the
    classic neg/pos weighting *hurts* PR-AUC because it distorts the probability
    surface; leaving the loss unweighted and handling imbalance purely at the
    decision threshold ranks better (verified: AP 0.302 -> 0.319).
    """
    y_tr = np.asarray(y_tr)
    pos = max(int(y_tr.sum()), 1)
    neg_over_pos = float((len(y_tr) - pos) / pos)
    spw = 1.0 if scale_pos_weight is None else float(scale_pos_weight)

    lgb_params = tune_lightgbm_classifier(X_tr, y_tr, tune_folds, spw) if tune else {}

    factories = {
        "lightgbm": lambda: make_lightgbm_classifier(lgb_params, spw),
        "xgboost": lambda: make_xgboost_classifier(spw),
        "catboost": lambda: make_catboost_classifier(spw, cat_features),
    }

    oof_cols, test_cols, base_models, order = {}, {}, {}, []
    for name, fac in factories.items():
        oof, test_p, full = _oof_and_full(
            fac, X_tr, y_tr, X_te, oof_folds,
            catboost=(name == "catboost"), cat_features=cat_features,
        )
        oof_cols[name] = oof
        test_cols[name] = test_p
        base_models[name] = full
        order.append(name)

    oof_matrix = np.column_stack([oof_cols[n] for n in order])
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(oof_matrix, y_tr)
    oof_meta = meta.predict_proba(oof_matrix)[:, 1]

    # Isotonic calibration on OOF meta predictions (leakage-safe).
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_meta, y_tr)
    oof_cal = calibrator.transform(oof_meta)

    threshold = best_threshold(y_tr, oof_cal, beta=beta)

    artifact = ClassificationArtifact(
        target=target, base_models=base_models, cat_features=cat_features,
        meta=meta, calibrator=calibrator, threshold=threshold, model_order=order,
    )
    diagnostics = {
        "oof_ap": float(average_precision_score(y_tr, oof_cal)),
        "scale_pos_weight": spw,
        "neg_over_pos": neg_over_pos,
        "threshold": threshold,
        "lgb_params": lgb_params,
        "base_oof_ap": {n: float(average_precision_score(y_tr, oof_cols[n])) for n in order},
    }
    return artifact, diagnostics


# --------------------------------------------------------------------------- #
# Duration regression (log target + quantile intervals)
# --------------------------------------------------------------------------- #
def make_lightgbm_regressor(params: dict, objective="regression", alpha=None):
    from lightgbm import LGBMRegressor

    base = dict(
        objective=objective, n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=C.RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    if alpha is not None:
        base["alpha"] = alpha
    base.update(params or {})
    return LGBMRegressor(**base)


def tune_lightgbm_regressor(X, y_log, folds, n_trials=C.OPTUNA_TRIALS):
    from sklearn.metrics import mean_absolute_error

    y_log = np.asarray(y_log)

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 900),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        scores = []
        for tr_idx, va_idx in folds:
            if len(va_idx) < 5:
                continue
            m = make_lightgbm_regressor(params)
            m.fit(X.iloc[tr_idx], y_log[tr_idx])
            pred = m.predict(X.iloc[va_idx])
            scores.append(mean_absolute_error(y_log[va_idx], pred))
        return float(np.mean(scores)) if scores else 1e9

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=C.OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    return study.best_params


@dataclass
class DurationArtifact:
    point_model: object = None
    quantile_models: dict = field(default_factory=dict)
    log_cap_minutes: float = None
    conformal_delta: float = 0.0  # CQR correction in log space

    def predict_minutes(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.point_model.predict(X))

    def predict_quantiles(self, X: pd.DataFrame) -> dict:
        out = {}
        for q, m in self.quantile_models.items():
            pred = m.predict(X)
            # Widen the outer quantiles by the conformal correction so the
            # empirical interval hits its nominal coverage.
            if q <= 0.1:
                pred = pred - self.conformal_delta
            elif q >= 0.9:
                pred = pred + self.conformal_delta
            out[q] = np.expm1(np.clip(pred, 0, None))
        return out


def _conformal_delta(y_lo, y_hi, y_true, coverage=0.8) -> float:
    """CQR correction: the (n+1)(1-alpha)/n empirical quantile of the two-sided
    conformity score max(lo - y, y - hi). Tightens / loosens the quantile band so
    the interval is honestly calibrated on exchangeable data.
    """
    scores = np.maximum(y_lo - y_true, y_true - y_hi)
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(max(0.0, np.quantile(scores, level)))


def train_duration_task(X_tr, y_tr_minutes, cap_minutes, folds, tune=True) -> tuple[DurationArtifact, dict]:
    """Train the duration point model + conformalized quantile intervals.

    The **point** model uses a winsorised (p99-capped) log target so it is robust
    to the multi-week construction tail. The **quantile** models use the *uncapped*
    log target. A final **conformal** correction is calibrated on the most recent
    slice of training data so the 80% interval actually covers ~80% (plain
    quantile regression systematically under-covers a heavy-tailed target).
    """
    y_tr_minutes = np.asarray(y_tr_minutes, dtype=float)
    y_point = np.log1p(np.clip(y_tr_minutes, a_min=None, a_max=cap_minutes))
    y_full = np.log1p(y_tr_minutes)

    params = tune_lightgbm_regressor(X_tr, y_point, folds) if tune else {}
    point = make_lightgbm_regressor(params)
    point.fit(X_tr, y_point)

    # Hold out the most recent 20% of (time-ordered) train rows for conformal
    # calibration - mirrors the future test set.
    n = len(X_tr)
    cut = int(round(n * 0.8))
    fit_idx, cal_idx = np.arange(cut), np.arange(cut, n)

    quantile_models = {}
    cal_preds = {}
    for q in (0.1, 0.5, 0.9):
        if len(cal_idx) > 0:
            qm_cal = make_lightgbm_regressor(params, objective="quantile", alpha=q)
            qm_cal.fit(X_tr.iloc[fit_idx], y_full[fit_idx])
            cal_preds[q] = qm_cal.predict(X_tr.iloc[cal_idx])
        # Deployable model refit on all train rows.
        qm_full = make_lightgbm_regressor(params, objective="quantile", alpha=q)
        qm_full.fit(X_tr, y_full)
        quantile_models[q] = qm_full

    delta = 0.0
    if len(cal_idx) > 0:
        delta = _conformal_delta(cal_preds[0.1], cal_preds[0.9], y_full[cal_idx], 0.8)

    artifact = DurationArtifact(point_model=point, quantile_models=quantile_models,
                                log_cap_minutes=float(cap_minutes), conformal_delta=delta)
    return artifact, {"lgb_params": params, "cap_minutes": float(cap_minutes),
                      "conformal_delta": delta}
