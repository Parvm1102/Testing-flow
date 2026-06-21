"""Inference service: load the trained Gridlock artifacts once and expose a
single-event scoring API plus batch helpers used by the precompute script.

Four predictions are produced for every event:
  * road-closure probability        (T1, src closure_model)
  * high-priority probability        (T2, src priority_model)
  * expected duration + 80% interval (T3, src duration_model)
  * chronic-hotspot risk             (T4, standalone hotspot_model)

A fifth, derived output - the **manpower level** (high / medium / low) - is a
transparent blend of the closure, priority and duration signals. It already
exists inside ``src.recommend`` (``minimal/low/medium/high`` + an officer
count); here we surface it and collapse ``minimal -> low`` for a clean 3-level
display.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Make the existing research code importable (repo root holds `src/` and
# `hotspot_model.py`). app/backend/inference.py -> app/backend -> app -> ROOT.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The 46-column raw schema the pipelines expect. A request fills the report-time
# fields; everything else (labels, post-event timestamps, ids) stays empty.
RAW_COLUMNS = [
    "id", "event_type", "latitude", "longitude", "endlatitude", "endlongitude",
    "address", "end_address", "event_cause", "requires_road_closure",
    "start_datetime", "end_datetime", "status", "authenticated",
    "modified_datetime", "map_file", "direction", "description", "veh_type",
    "veh_no", "corridor", "priority", "cargo_material", "reason_breakdown",
    "age_of_truck", "created_date", "route_path", "client_id", "created_by_id",
    "last_modified_by_id", "assigned_to_police_id", "citizen_accident_id",
    "comment", "police_station", "meta_data", "kgid", "resolved_at_address",
    "resolved_at_latitude", "resolved_at_longitude", "closed_by_id",
    "closed_datetime", "resolved_by_id", "resolved_datetime", "gba_identifier",
    "zone", "junction",
]

# Categorical fields a user can actually choose at report time (drives dropdowns).
USER_CATEGORICAL_FIELDS = [
    "event_type", "event_cause", "veh_type", "direction", "corridor",
    "zone", "junction", "police_station", "reason_breakdown",
    "cargo_material", "authenticated",
]

LAT_RANGE = (12.6, 13.4)
LON_RANGE = (77.2, 77.9)


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and NaN into JSON-native types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT or (value is None):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def three_level_manpower(tier: str) -> str:
    """Collapse the internal 4-tier manpower into a 3-level demo display."""
    return {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}.get(
        str(tier).lower(), "low"
    )


class InferenceService:
    """Loads every artifact once; reused across requests."""

    def __init__(self) -> None:
        # Heavy imports happen here, not at module import time.
        from src.predict import GridlockPredictor
        import joblib
        import hotspot_model as HM

        self.predictor = GridlockPredictor()

        self.HM = HM
        self.hotspot_bundle = joblib.load(HM.BUNDLE_PATH)
        self.hotspot_history = pd.read_parquet(HM.HISTORY_PATH)
        # Cache the dropdown vocabulary so /api/options is instant.
        self._options_cache: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------ #
    # Vocabulary for form dropdowns (exact categories the models learned)
    # ------------------------------------------------------------------ #
    def category_options(self) -> dict[str, list[str]]:
        if self._options_cache is not None:
            return self._options_cache

        prep_full = self.predictor.prep_full
        prep_priority = self.predictor.prep_priority
        options: dict[str, list[str]] = {}
        for field in USER_CATEGORICAL_FIELDS:
            cats: list[str] = []
            for prep in (prep_full, prep_priority):
                learned = getattr(prep, "_cat_categories", {}).get(field)
                if learned is not None:
                    cats = [str(c) for c in learned]
                    break
            options[field] = sorted({c for c in cats if c and c.lower() != "nan"})

        # event_cause: guarantee the canonical family keys are present/ordered.
        from src import config as C
        canonical = list(C.EVENT_FAMILY_MAP.keys())
        learned_cause = set(options.get("event_cause", []))
        merged = [c for c in canonical if c in learned_cause] + sorted(
            learned_cause - set(canonical)
        )
        if merged:
            options["event_cause"] = merged
        # authenticated reads cleaner as yes/no in the UI.
        options["authenticated"] = ["yes", "no"]
        self._options_cache = options
        return options

    # ------------------------------------------------------------------ #
    # Row assembly
    # ------------------------------------------------------------------ #
    def _raw_frame(self, payloads: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for i, payload in enumerate(payloads):
            row = {c: None for c in RAW_COLUMNS}
            for key, val in payload.items():
                if key in RAW_COLUMNS and val is not None and val != "":
                    row[key] = val
            if row.get("id") in (None, ""):
                row["id"] = f"REQ{i:06d}"
            # If no advance-notice timestamp given, the event is reported as it
            # starts (created_date == start_datetime -> zero lead time).
            if row.get("created_date") in (None, "") and row.get("start_datetime"):
                row["created_date"] = row["start_datetime"]
            rows.append(row)
        return pd.DataFrame(rows, columns=RAW_COLUMNS)

    # ------------------------------------------------------------------ #
    # Hotspot scoring (reuses preloaded bundle + history; no per-call IO)
    # ------------------------------------------------------------------ #
    def _hotspot_scores(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Return raw_df rows with `hotspot_risk` + `hotspot_flag` columns.

        New events are concatenated onto the saved history snapshot so their
        causal features see the same accumulated past the model trained on.
        """
        HM = self.HM
        new = raw_df.copy()
        # Type the report-time columns the way HM.load_and_clean would.
        new["latitude"] = pd.to_numeric(new["latitude"], errors="coerce")
        new["longitude"] = pd.to_numeric(new["longitude"], errors="coerce")
        new["age_of_truck"] = pd.to_numeric(new.get("age_of_truck"), errors="coerce")
        for col in ("start_datetime", "created_date"):
            new[col] = pd.to_datetime(new[col], utc=True, errors="coerce")
        new["requires_road_closure"] = 0  # own (future) label is irrelevant here
        new["order_time"] = new["created_date"].fillna(new["start_datetime"])

        valid = new["latitude"].notna() & new["longitude"].notna() & new["order_time"].notna()
        result = raw_df.copy().reset_index(drop=True)
        result["hotspot_risk"] = np.nan
        result["hotspot_flag"] = 0
        if not valid.any():
            return result

        new_valid = new[valid].copy()
        new_valid["__is_new__"] = True
        history = self.hotspot_history.copy()
        history["__is_new__"] = False
        combined = pd.concat([history, new_valid], ignore_index=True)
        combined["order_time"] = pd.to_datetime(combined["order_time"], utc=True)
        combined = combined.sort_values("order_time").reset_index(drop=True)

        feats, _, _ = HM.assemble_features(combined)
        feats = HM.apply_category_dtypes(feats, self.hotspot_bundle["cat_dtypes"])[
            self.hotspot_bundle["feature_cols"]
        ]
        mask = combined["__is_new__"].to_numpy()
        raw = self.hotspot_bundle["model"].predict_proba(feats[mask])[:, 1]
        prob = self.hotspot_bundle["isotonic"].predict(raw)
        thr = float(self.hotspot_bundle["threshold"])

        result.loc[valid.values, "hotspot_risk"] = prob
        result.loc[valid.values, "hotspot_flag"] = (prob >= thr).astype(int)
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score one event and return all predictions + recommendations."""
        # Validate the few hard requirements up front for a clean error.
        lat = pd.to_numeric(payload.get("latitude"), errors="coerce")
        lon = pd.to_numeric(payload.get("longitude"), errors="coerce")
        if pd.isna(lat) or not (LAT_RANGE[0] <= float(lat) <= LAT_RANGE[1]):
            raise ValueError(
                f"latitude must be a Bengaluru coordinate in {LAT_RANGE}, got {payload.get('latitude')!r}"
            )
        if pd.isna(lon) or not (LON_RANGE[0] <= float(lon) <= LON_RANGE[1]):
            raise ValueError(
                f"longitude must be a Bengaluru coordinate in {LON_RANGE}, got {payload.get('longitude')!r}"
            )
        if not payload.get("start_datetime"):
            raise ValueError("start_datetime is required")
        if pd.isna(pd.to_datetime(payload.get("start_datetime"), utc=True, errors="coerce")):
            raise ValueError(f"start_datetime could not be parsed: {payload.get('start_datetime')!r}")

        raw_df = self._raw_frame([payload])
        out = self.predictor.predict_frame(raw_df)
        if len(out) == 0:
            raise ValueError("Event could not be scored (failed cleaning).")
        rec = out.iloc[0].to_dict()

        hot = self._hotspot_scores(raw_df).iloc[0]
        rec["hotspot_risk"] = hot["hotspot_risk"]
        rec["hotspot_flag"] = int(hot["hotspot_flag"])
        rec["manpower_level"] = three_level_manpower(rec.get("manpower_tier", "low"))
        return _json_safe(rec)

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Score many raw events at once (used by precompute). Returns a frame
        joined on `event_id` with main predictions + hotspot risk."""
        main = self.predictor.predict_frame(raw_df)
        hot = self._hotspot_scores(raw_df)
        # Align hotspot risk back by id.
        if "id" in raw_df.columns:
            hot_idx = hot.copy()
            hot_idx["event_id"] = raw_df["id"].values
            main = main.merge(
                hot_idx[["event_id", "hotspot_risk", "hotspot_flag"]],
                on="event_id", how="left",
            )
        else:
            main["hotspot_risk"] = hot["hotspot_risk"].values
            main["hotspot_flag"] = hot["hotspot_flag"].values
        main["manpower_level"] = main["manpower_tier"].map(three_level_manpower)
        return main


_SERVICE: InferenceService | None = None


def get_service() -> InferenceService:
    """Lazy singleton so importing this module stays cheap."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = InferenceService()
    return _SERVICE
