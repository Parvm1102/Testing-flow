"""Target engineering for the three prediction tasks.

T1  y_road_closure   : bool   -> barricading / diversion need (imbalanced ~7%)
T2  y_high_priority  : bool   -> manpower tier (priority == "High")
T3  y_duration_min   : float  -> impact duration in minutes (regression)

Duration is derived from post-event timestamps (the only place it exists) but
those timestamps are NEVER used as model features. The duration label is built
by coalescing resolved -> closed -> end (most reliable first). Rows that were
auto-closed by the nightly batch job, are still active, or have a non-positive
duration are marked invalid and excluded from regression training only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _coalesced_end(df: pd.DataFrame) -> pd.Series:
    """Best available clearance timestamp: resolved > closed > end."""
    cols = ["resolved_datetime", "closed_datetime", "end_datetime"]
    present = [c for c in cols if c in df.columns]
    return df[present].bfill(axis=1).iloc[:, 0]


def build_targets(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    df = df.copy()

    # ---- T1: road closure (barricading / diversion) --------------------- #
    df[C.TARGET_CLOSURE] = df["requires_road_closure"].fillna(False).astype(int)

    # ---- T2: high priority (manpower tier) ------------------------------ #
    df[C.TARGET_PRIORITY] = (
        df["priority"].astype(str).str.strip().str.lower().eq("high").astype(int)
    )

    # ---- T3: impact duration (minutes) ---------------------------------- #
    end = _coalesced_end(df)
    duration = (end - df["start_datetime"]).dt.total_seconds() / 60.0

    valid = (
        duration.gt(C.DURATION_MIN_MINUTES)
        & (~df["auto_resolved_flag"].fillna(False))
        & df["status"].astype(str).str.lower().ne("active")
    )
    df[C.TARGET_DURATION] = np.where(valid, duration, np.nan)
    df["duration_valid"] = valid.astype(bool)

    if save:
        df.to_parquet(C.CLEAN_PARQUET, index=False)
    return df


def winsorized_log_duration(y_min: pd.Series, upper_cap: float | None = None):
    """Return (log1p target, fitted upper cap). Cap is computed ONLY on the
    values passed in (the trainer passes the training split to avoid leakage).
    """
    y = y_min.dropna()
    if upper_cap is None:
        upper_cap = float(np.quantile(y, C.DURATION_WINSOR_UPPER_Q))
    capped = np.clip(y_min, a_min=None, a_max=upper_cap)
    return np.log1p(capped), upper_cap


if __name__ == "__main__":  # pragma: no cover
    d = pd.read_parquet(C.CLEAN_PARQUET)
    d = build_targets(d)
    print("closure positive rate:", d[C.TARGET_CLOSURE].mean().round(4))
    print("high-priority rate   :", d[C.TARGET_PRIORITY].mean().round(4))
    print("duration valid rows  :", int(d["duration_valid"].sum()))
    print(d.loc[d.duration_valid, C.TARGET_DURATION].describe().round(1))
