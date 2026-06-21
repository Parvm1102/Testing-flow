"""Tabular feature engineering (leakage-safe, creation-time information only).

All features here are computable at the moment an event is reported. Features
that require knowledge of how the event unfolded (resolution time, status,
assigned officers) are deliberately excluded - see config.LEAKAGE_COLUMNS.

History-dependent features (hotspot density, recurrence counts) are computed in
a strictly causal way: each row only sees events that were reported *before* it
in chronological order, so no future information leaks backward.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_PIN_RE = re.compile(r"(?:Pin-)?(\d{6})")


def _extract_pincode(address):
    if not isinstance(address, str):
        return np.nan
    m = _PIN_RE.search(address.replace(" ", ""))
    return m.group(1) if m else np.nan


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #
def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use IST (UTC+5:30); Bengaluru operational clock drives traffic patterns.
    start_ist = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    df["start_hour"] = start_ist.dt.hour
    df["start_dow"] = start_ist.dt.dayofweek
    df["start_month"] = start_ist.dt.month
    df["start_weekofyear"] = start_ist.dt.isocalendar().week.astype(int)
    df["start_day"] = start_ist.dt.day
    df["is_weekend"] = (df["start_dow"] >= 5).astype(int)
    df["is_morning_peak"] = df["start_hour"].between(8, 11).astype(int)
    df["is_evening_peak"] = df["start_hour"].between(17, 21).astype(int)
    df["is_night"] = ((df["start_hour"] >= 22) | (df["start_hour"] <= 5)).astype(int)

    # Advance notice: planned events are logged hours/days before they start.
    if "created_date" in df.columns:
        lead = (df["start_datetime"] - df["created_date"]).dt.total_seconds() / 3600.0
        # Negative lead (created after start) -> treat as no notice (0).
        df["lead_time_hours"] = lead.clip(lower=0).fillna(0.0)
        df["has_advance_notice"] = (lead > 0.5).fillna(False).astype(int)
    else:
        df["lead_time_hours"] = 0.0
        df["has_advance_notice"] = 0

    # Cyclical encodings.
    df["hour_sin"] = np.sin(2 * np.pi * df["start_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["start_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["start_dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["start_dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["start_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["start_month"] / 12)
    return df


def _spatial_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    # NOTE: end-point coordinates and route_path describe the closure/diversion
    # SEGMENT that is drawn as a consequence of the closure decision. They leak
    # the target (see config.LEAKAGE_COLUMNS) and are deliberately NOT used.
    # Only the report-time event location is kept.
    df["pincode"] = df["address"].apply(_extract_pincode)
    # Target-free spatial binning so trees can generalise across nearby points.
    # KMeans on coordinates is unsupervised (no label) -> not leakage. The fitted
    # model is persisted so inference assigns the SAME bins (cluster ids would
    # otherwise permute on every refit and silently break the categorical).
    coords = df[["latitude", "longitude"]].to_numpy()
    try:
        import joblib
        from sklearn.cluster import MiniBatchKMeans

        if training:
            k = min(C.GEO_CLUSTERS, max(2, len(df) // 50))
            km = MiniBatchKMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=3)
            km.fit(coords)
            joblib.dump(km, C.GEO_KMEANS_PATH)
        else:
            km = joblib.load(C.GEO_KMEANS_PATH)
        df["geo_cluster"] = km.predict(coords).astype(int).astype(str)
    except Exception:
        df["geo_cluster"] = "0"
    return df


def _causal_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly causal recurrence / hotspot features (past events only).

    Rows are already sorted by ``order_time`` (report time) in cleaning, so a
    simple expanding ``cumcount`` over location/cause groups counts only events
    that were reported earlier.
    """
    # Coarse geocell (~1.1 km) for hotspot aggregation.
    cell = (
        df["latitude"].round(2).astype(str) + "_" + df["longitude"].round(2).astype(str)
    )
    df["_geocell"] = cell
    # How many events previously reported in this geocell.
    df["hist_hotspot_count"] = df.groupby("_geocell").cumcount()
    # Normalised local density (per elapsed day in dataset).
    elapsed_days = (
        (df["order_time"] - df["order_time"].min()).dt.total_seconds() / 86400.0
    ).clip(lower=1.0)
    df["loc_event_density"] = df["hist_hotspot_count"] / elapsed_days

    # Same location + same cause history (recurring construction/festival sites).
    df["same_loc_cause_hist"] = df.groupby(["_geocell", "event_cause"]).cumcount()

    # Same calendar day + location report burst (event-magnitude proxy).
    day = df["start_datetime"].dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
    df["same_day_loc_reports"] = df.groupby([day, "_geocell"]).cumcount()
    df.drop(columns=["_geocell"], inplace=True)
    return df


def _categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    cause_norm = df["event_cause"].astype(str).str.strip().str.lower()
    df["event_family"] = cause_norm.map(C.EVENT_FAMILY_MAP).fillna(C.EVENT_FAMILY_DEFAULT)
    df["veh_type_missing"] = df["veh_type"].isna().astype(int)
    # Normalise authenticated to string category for the encoder.
    if "authenticated" in df.columns:
        df["authenticated"] = df["authenticated"].map({True: "yes", False: "no"}).fillna("unknown")
    return df


def _lexicon_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df[C.TEXT_COLUMN].fillna("").astype(str).str.lower()
    df["desc_has_text"] = (text.str.len() > 0).astype(int)
    df["desc_missing"] = (text.str.len() == 0).astype(int)
    df["desc_len"] = text.str.len()
    df["desc_word_count"] = text.str.split().apply(len)
    # Kannada Unicode block presence.
    df["desc_is_kannada"] = text.str.contains(r"[\u0c80-\u0cff]", regex=True).astype(int)
    for feat, terms in C.LEXICON.items():
        pattern = "|".join(re.escape(t) for t in terms)
        df[feat] = text.str.contains(pattern, regex=True).astype(int)
    return df


def _causal_target_features(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Past-only, leakage-safe target encodings.

    For each grouping key we build a smoothed *running* mean of the closure label
    over events reported **earlier in time** (``shift(1)`` excludes the row's own
    label). Low-count groups are shrunk toward the running global rate
    (empirical-Bayes smoothing). This gives the rare closure target far more
    signal than the static category alone and, because it is recomputed from
    accumulated history, it tracks drift.

    At inference ``history`` (the accumulated training frame) is prepended so a
    new event sees the same historical rates the model trained on; its own
    (unknown) label is never used.
    """
    keys = [k for k in C.CLOSURE_RATE_KEYS if k in df.columns]
    rate_cols = list(C.CLOSURE_RATE_KEYS.values()) + ["dur_recent_level"]
    if C.TARGET_CLOSURE not in df.columns or not keys:
        for col in rate_cols:
            df[col] = np.nan
        return df

    needed = keys + [C.TARGET_CLOSURE, C.TARGET_DURATION, "duration_valid", "order_time"]
    cur = df.reindex(columns=[c for c in needed if c in df.columns]).copy()
    cur["_seg"] = 1
    cur["_pos"] = np.arange(len(df))

    if history is not None and len(history):
        hist = history.reindex(columns=cur.columns.drop(["_seg", "_pos"])).copy()
        hist["_seg"] = 0
        hist["_pos"] = -1
        base = pd.concat([hist, cur], ignore_index=True)
    else:
        base = cur.reset_index(drop=True)

    base = base.sort_values("order_time", kind="stable").reset_index(drop=True)

    # Closure label as float; unknown (inference) labels -> 0 so they never leak
    # into a later row's running sum (shift(1) already excludes the row itself).
    yc = pd.to_numeric(base[C.TARGET_CLOSURE], errors="coerce").fillna(0.0)
    global_prior = yc.expanding().mean().shift(1).fillna(yc.mean())
    m = C.CAUSAL_TARGET_SMOOTHING
    rates = {}
    for key in keys:
        grp = yc.groupby(base[key].astype("string"), dropna=False)
        csum = grp.cumsum().shift(1).fillna(0.0)
        ccnt = grp.cumcount().shift(1).fillna(0.0)
        rates[C.CLOSURE_RATE_KEYS[key]] = (csum + m * global_prior) / (ccnt + m)

    # Ambient duration level: rolling mean of the last-N resolved log-durations.
    logd = np.log1p(pd.to_numeric(base[C.TARGET_DURATION], errors="coerce"))
    valid = (base["duration_valid"].fillna(False).astype(bool)
             if "duration_valid" in base else logd.notna())
    sv = logd.where(valid).dropna()
    roll = sv.rolling(C.DUR_RECENT_WINDOW, min_periods=5).mean().shift(1)
    recent = pd.Series(np.nan, index=base.index)
    recent.loc[sv.index] = roll
    recent = recent.ffill().fillna(sv.mean() if len(sv) else 0.0)

    mask = base["_seg"].eq(1).to_numpy()
    order = np.argsort(base.loc[mask, "_pos"].to_numpy())
    for col, series in rates.items():
        df[col] = series.to_numpy()[mask][order]
    df["dur_recent_level"] = recent.to_numpy()[mask][order]
    return df


def _numeric_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df["age_of_truck"] = pd.to_numeric(df["age_of_truck"], errors="coerce")
    return df


def build_features(df: pd.DataFrame, save: bool = True, training: bool = True,
                   history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run all feature blocks. Input must already have targets + order_time.

    ``training`` controls whether the spatial KMeans is fit (and persisted) or
    loaded. ``history`` is the accumulated training frame, prepended for the
    causal target encodings so inference matches training.
    """
    df = df.copy()
    df = _temporal_features(df)
    df = _spatial_features(df, training=training)
    df = _causal_history_features(df)
    df = _categorical_features(df)
    df = _lexicon_features(df)
    df = _causal_target_features(df, history=history)
    df = _numeric_cleanup(df)
    if save:
        df.to_parquet(C.FEATURES_PARQUET, index=False)
    return df


def history_columns() -> list[str]:
    """Columns the causal target encodings need from accumulated history."""
    keys = list(C.CLOSURE_RATE_KEYS)
    extra = [C.TARGET_CLOSURE, C.TARGET_DURATION, "duration_valid", "order_time"]
    return keys + extra


def save_history(df: pd.DataFrame) -> None:
    """Persist the minimal labeled history needed to reproduce, at inference
    time, the same past-only target-rate encodings seen during training.
    """
    cols = [c for c in history_columns() if c in df.columns]
    df[cols].to_parquet(C.HISTORY_PARQUET, index=False)


def load_history() -> pd.DataFrame | None:
    """Load accumulated history for inference; ``None`` if not yet persisted."""
    if C.HISTORY_PARQUET.exists():
        return pd.read_parquet(C.HISTORY_PARQUET)
    return None


if __name__ == "__main__":  # pragma: no cover
    from .targets import build_targets

    d = pd.read_parquet(C.CLEAN_PARQUET)
    if C.TARGET_CLOSURE not in d.columns:
        d = build_targets(d, save=False)
    d = build_features(d)
    feats = [c for c in C.NUMERIC_FEATURES if c in d.columns]
    print("rows:", len(d), "| numeric features present:", len(feats))
    print(d[["event_family", "geo_cluster", "hist_hotspot_count",
             "lead_time_hours", "lex_festival", "desc_is_kannada"]].describe(include="all").round(2))
