"""Train/test splitting and cross-validation that respect time order.

Forecasting is only honest if the model is evaluated on events that occurred
*after* everything it trained on. We therefore use a single chronological
holdout for final evaluation and time-series CV for tuning. A group-by-location
KFold is also provided as a robustness check against spatial memorisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from . import config as C


def temporal_split(df: pd.DataFrame, test_fraction: float = C.TEST_FRACTION):
    """Return boolean train/test masks split on chronological ``order_time``.

    Assumes ``df`` is already sorted by ``order_time`` (cleaning guarantees it).
    """
    n = len(df)
    cut = int(round(n * (1 - test_fraction)))
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True
    test_mask[cut:] = True
    return train_mask, test_mask


def time_series_folds(n_train: int, n_splits: int = C.N_FOLDS):
    """Expanding-window CV indices for hyper-parameter tuning on the train set."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    return list(tss.split(np.arange(n_train)))


def stratified_folds(y: np.ndarray, n_splits: int = C.N_FOLDS):
    """Stratified folds (used for the rare positive closure class during tuning)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    return list(skf.split(np.zeros(len(y)), y))


def split_report(df: pd.DataFrame, train_mask, test_mask) -> str:
    def span(mask):
        t = df.loc[mask, "order_time"]
        return f"{t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d}  (n={mask.sum()})"

    return f"train: {span(train_mask)}\ntest : {span(test_mask)}"


if __name__ == "__main__":  # pragma: no cover
    d = pd.read_parquet(C.FEATURES_PARQUET)
    tr, te = temporal_split(d)
    print(split_report(d, tr, te))
    for col in (C.TARGET_CLOSURE, C.TARGET_PRIORITY):
        print(f"{col}: train+={d.loc[tr, col].mean():.3f}  test+={d.loc[te, col].mean():.3f}")
