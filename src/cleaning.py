"""Data cleaning: NULL normalisation, type coercion, timestamp parsing,
coordinate sanity fixes, de-duplication and auto-resolution flagging.

Nothing here builds features or targets; it only produces a typed, sane frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data_loading import load_raw


def _strip_null_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the dataset's many textual NULL spellings with real NaN."""
    return df.map(
        lambda v: np.nan if isinstance(v, str) and v.strip() in C.NULL_TOKENS else v
    )


def _parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for col in C.DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["latitude", "longitude", "endlatitude", "endlongitude", "age_of_truck"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    if "requires_road_closure" in df.columns:
        df["requires_road_closure"] = (
            df["requires_road_closure"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
        )
    if "authenticated" in df.columns:
        df["authenticated"] = (
            df["authenticated"].astype(str).str.lower().map({"yes": True, "no": False})
        )
    return df


def _fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Sentinel 0/placeholder coordinates -> NaN; keep only plausible Bengaluru.

    End coordinates use ``0`` as a "no end point" sentinel. Main coordinates
    occasionally fall outside a sane bounding box (data entry noise).
    """
    for col in ["endlatitude", "endlongitude"]:
        if col in df.columns:
            df.loc[df[col].abs() < 1e-6, col] = np.nan

    # Bengaluru metropolitan bounding box (generous).
    lat_ok = df["latitude"].between(12.6, 13.4)
    lon_ok = df["longitude"].between(77.2, 77.9)
    df.loc[~(lat_ok), "latitude"] = np.nan
    df.loc[~(lon_ok), "longitude"] = np.nan

    if {"endlatitude", "endlongitude"}.issubset(df.columns):
        elat_ok = df["endlatitude"].between(12.6, 13.4)
        elon_ok = df["endlongitude"].between(77.2, 77.9)
        df.loc[~elat_ok, "endlatitude"] = np.nan
        df.loc[~elon_ok, "endlongitude"] = np.nan
    return df


def _flag_auto_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose resolution/modification looks like a nightly batch job.

    Many records are auto-closed at minute/second ``35:47`` or ``05:46`` which
    are clearly system timestamps rather than the true clearance time. These
    make the duration label unreliable and are flagged so targets.py can decide
    how to treat them.
    """
    suspicious = pd.Series(False, index=df.index)
    for col in ["modified_datetime", "resolved_datetime", "closed_datetime"]:
        if col in df.columns:
            ts = df[col]
            mark = (ts.dt.minute == 35) & (ts.dt.second.between(47, 48))
            mark |= (ts.dt.second == 46) & (ts.dt.minute.isin([5, 35]))
            suspicious = suspicious | mark.fillna(False)
    df["auto_resolved_flag"] = suspicious
    return df


def clean(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """Run the full cleaning pipeline and (optionally) persist a parquet."""
    if df is None:
        df = load_raw()

    df = _strip_null_tokens(df)
    df = _parse_datetimes(df)
    df = _coerce_numeric(df)
    df = _coerce_booleans(df)
    df = _fix_coordinates(df)

    # Drop exact duplicate event ids (keep the first occurrence).
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="first")

    # Require a usable start time and location to be a modelable event.
    df = df[df["start_datetime"].notna()].copy()
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    df = _flag_auto_resolution(df)

    # Order chronologically by report/creation time -> enables temporal splits.
    df["order_time"] = df["created_date"].fillna(df["start_datetime"])
    df = df.sort_values("order_time").reset_index(drop=True)

    if save:
        df.to_parquet(C.CLEAN_PARQUET, index=False)
    return df


if __name__ == "__main__":  # pragma: no cover
    out = clean()
    print("clean shape:", out.shape)
    print(out[["event_type", "event_cause", "requires_road_closure", "priority"]].describe(include="all"))
