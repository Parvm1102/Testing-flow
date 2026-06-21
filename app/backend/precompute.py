"""Offline precompute: score every historical event once and write the static
JSON the Map, Top-Areas and Models tabs read at runtime.

Outputs (in app/backend/generated/):
  * options.json  - dropdown vocab + police-station centroids + map bounds
  * areas.json    - per police-station aggregates (table + map markers)
  * map.json      - ~110 m hotspot cells + per-event points for heat layers
  * metrics.json  - combined model metrics for the Models tab

Run from the repo root with the venv active:
    python -m app.backend.precompute
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.inference import _json_safe, get_service, three_level_manpower  # noqa: E402

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

UNKNOWN_STATION = "Unknown / No Station"
GRID_DECIMALS = 3  # ~110 m cell, matches the hotspot model's location unit


def _write(name: str, obj) -> None:
    path = GENERATED_DIR / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(obj), fh, ensure_ascii=False, separators=(",", ":"))
    size_kb = path.stat().st_size / 1024
    print(f"  wrote {name}  ({size_kb:.0f} KB)")


def build_master(service) -> pd.DataFrame:
    """One row per scored event with geo, actual labels and all predictions."""
    from src.cleaning import clean
    from src.data_loading import load_raw
    from src.targets import build_targets

    print("[precompute] loading + scoring all events (transformer embeddings)...")
    raw = load_raw()

    # Geo + admin + actual labels come from the hotspot loader (keeps id and all
    # raw columns, types coordinates) so we can group/plot reliably.
    HM = service.HM
    geo = HM.load_and_clean()  # id, latitude, longitude, police_station, junction, zone, event_cause, requires_road_closure(1/0), priority

    # Main predictions (closure / priority / duration / manpower).
    main = service.predictor.predict_frame(raw)
    main = main.rename(columns={"event_id": "id"})

    # Hotspot risk computed causally over the FULL frame (each event sees only
    # its past) - the faithful way to replay the model, no history double-count.
    feats, _, _ = HM.assemble_features(geo)
    feats = HM.apply_category_dtypes(feats, service.hotspot_bundle["cat_dtypes"])[
        service.hotspot_bundle["feature_cols"]
    ]
    raw_proba = service.hotspot_bundle["model"].predict_proba(feats)[:, 1]
    risk = service.hotspot_bundle["isotonic"].predict(raw_proba)
    thr = float(service.hotspot_bundle["threshold"])
    geo = geo.copy()
    geo["hotspot_risk"] = risk
    geo["hotspot_flag"] = (risk >= thr).astype(int)

    # Actual clearance duration (heavy-tailed; only some rows are valid).
    tdf = build_targets(clean(raw, save=False), save=False)
    dur = tdf[["id", "y_duration_min", "duration_valid"]].copy()

    cols_geo = [
        "id", "latitude", "longitude", "police_station", "junction", "zone",
        "event_cause", "requires_road_closure", "priority", "hotspot_risk",
        "hotspot_flag",
    ]
    cols_main = [
        "id", "closure_probability", "high_priority_probability",
        "expected_duration_min", "officers_suggested", "manpower_tier",
    ]
    master = (
        geo[cols_geo]
        .merge(main[cols_main], on="id", how="inner")
        .merge(dur, on="id", how="left")
    )
    master["manpower_level"] = master["manpower_tier"].map(three_level_manpower)
    master["is_high_priority"] = (master["priority"].astype(str).str.lower() == "high").astype(int)
    master["requires_road_closure"] = pd.to_numeric(
        master["requires_road_closure"], errors="coerce"
    ).fillna(0)
    master["police_station"] = master["police_station"].fillna(UNKNOWN_STATION).replace(
        {"": UNKNOWN_STATION}
    )
    print(f"[precompute] master frame: {len(master):,} events")
    return master


def _top_causes(series: pd.Series, n: int = 3) -> list[str]:
    vc = series.dropna().astype(str)
    vc = vc[vc.str.lower() != "nan"]
    return vc.value_counts().head(n).index.tolist()


def aggregate_areas(master: pd.DataFrame) -> list[dict]:
    areas = []
    for name, g in master.groupby("police_station"):
        n = len(g)
        valid_dur = g.loc[g["duration_valid"] == True, "y_duration_min"] if "duration_valid" in g else pd.Series(dtype=float)
        closure_rate = float(g["requires_road_closure"].mean())
        pred_closure = float(g["closure_probability"].mean())
        high_pri_rate = float(g["is_high_priority"].mean())
        pred_high_pri = float(g["high_priority_probability"].mean())
        avg_hotspot = float(g["hotspot_risk"].mean())
        risk_score = round(
            100.0 * (0.45 * pred_closure + 0.35 * avg_hotspot + 0.20 * pred_high_pri), 1
        )
        level_counts = g["manpower_level"].value_counts()
        areas.append({
            "area": name,
            "n_events": int(n),
            "lat": round(float(g["latitude"].median()), 5),
            "lng": round(float(g["longitude"].median()), 5),
            "closure_rate": round(closure_rate, 4),
            "pred_closure_rate": round(pred_closure, 4),
            "high_priority_rate": round(high_pri_rate, 4),
            "pred_high_priority_rate": round(pred_high_pri, 4),
            "avg_duration_min": round(float(valid_dur.median()), 1) if len(valid_dur) else None,
            "pred_avg_duration_min": round(float(g["expected_duration_min"].median()), 1),
            "avg_hotspot_risk": round(avg_hotspot, 4),
            "chronic_count": int(g["hotspot_flag"].sum()),
            "chronic_rate": round(float(g["hotspot_flag"].mean()), 4),
            "avg_officers": round(float(g["officers_suggested"].mean()), 2),
            "manpower_high": int(level_counts.get("high", 0)),
            "manpower_medium": int(level_counts.get("medium", 0)),
            "manpower_low": int(level_counts.get("low", 0)),
            "risk_score": risk_score,
            "top_causes": _top_causes(g["event_cause"]),
        })
    areas.sort(key=lambda a: a["n_events"], reverse=True)
    return areas


def build_map(master: pd.DataFrame) -> dict:
    # ~110 m hotspot cells (the model's own location unit).
    m = master.dropna(subset=["latitude", "longitude"]).copy()
    m["glat"] = m["latitude"].round(GRID_DECIMALS)
    m["glng"] = m["longitude"].round(GRID_DECIMALS)
    cells = []
    for (glat, glng), g in m.groupby(["glat", "glng"]):
        count = len(g)
        max_risk = float(g["hotspot_risk"].max())
        if count < 2 and max_risk < 0.2:
            continue  # skip lonely low-risk cells to keep the layer crisp
        label = g["police_station"].mode().iat[0] if len(g["police_station"].mode()) else UNKNOWN_STATION
        junctions = g["junction"].dropna().astype(str)
        junctions = junctions[junctions.str.lower() != "nan"]
        cells.append({
            "lat": round(float(glat), 3),
            "lng": round(float(glng), 3),
            "count": int(count),
            "max_risk": round(max_risk, 4),
            "mean_risk": round(float(g["hotspot_risk"].mean()), 4),
            "chronic_count": int(g["hotspot_flag"].sum()),
            "closure_rate": round(float(g["requires_road_closure"].mean()), 4),
            "label": label,
            "junction": junctions.mode().iat[0] if len(junctions.mode()) else None,
            "top_cause": _top_causes(g["event_cause"], 1)[0] if _top_causes(g["event_cause"], 1) else None,
        })
    cells.sort(key=lambda c: c["max_risk"], reverse=True)

    # Per-event points for the heat layers (compact arrays: lat,lng,closure,hotspot,officers).
    pts = [
        [round(float(r.latitude), 4), round(float(r.longitude), 4),
         round(float(r.closure_probability), 3),
         round(float(r.hotspot_risk), 3) if pd.notna(r.hotspot_risk) else 0.0,
         int(r.officers_suggested)]
        for r in m.itertuples()
    ]
    bounds = {
        "min_lat": round(float(m["latitude"].min()), 4),
        "max_lat": round(float(m["latitude"].max()), 4),
        "min_lng": round(float(m["longitude"].min()), 4),
        "max_lng": round(float(m["longitude"].max()), 4),
        "center_lat": round(float(m["latitude"].median()), 4),
        "center_lng": round(float(m["longitude"].median()), 4),
    }
    print(f"[precompute] map: {len(cells):,} hotspot cells, {len(pts):,} points")
    return {
        "point_fields": ["lat", "lng", "closure_prob", "hotspot_risk", "officers"],
        "points": pts,
        "hotspot_cells": cells,
        "bounds": bounds,
    }


def build_options(service, master: pd.DataFrame, areas: list[dict]) -> dict:
    bounds = {
        "center_lat": round(float(master["latitude"].median()), 4),
        "center_lng": round(float(master["longitude"].median()), 4),
        "min_lat": round(float(master["latitude"].min()), 4),
        "max_lat": round(float(master["latitude"].max()), 4),
        "min_lng": round(float(master["longitude"].min()), 4),
        "max_lng": round(float(master["longitude"].max()), 4),
    }
    station_centroids = [
        {"name": a["area"], "lat": a["lat"], "lng": a["lng"], "n_events": a["n_events"]}
        for a in areas if a["area"] != UNKNOWN_STATION
    ]
    return {
        "categories": service.category_options(),
        "stations": station_centroids,
        "bounds": bounds,
    }


def build_metrics(master: pd.DataFrame) -> dict:
    from src import config as C

    def _load(path: Path):
        return json.loads(path.read_text()) if path.exists() else {}

    reports = _load(C.REPORTS_DIR / "metrics.json")
    hotspot = _load(ROOT / "hotspot_artifacts" / "hotspot_metrics.json")
    closure_best = _load(C.REPORTS_DIR / "closure_best_operating_points.json")

    dataset = {
        "n_events_scored": int(len(master)),
        "n_areas": int(master["police_station"].nunique()),
        "closure_base_rate": round(float(master["requires_road_closure"].mean()), 4),
        "high_priority_base_rate": round(float(master["is_high_priority"].mean()), 4),
        "chronic_rate": round(float(master["hotspot_flag"].mean()), 4),
        "date_span": "9 Nov 2023 - 8 Apr 2024 (~150 days)",
    }
    return {
        "dataset": dataset,
        "priority": reports.get("priority", {}),
        "closure": reports.get("closure", {}),
        "duration": reports.get("duration", {}),
        "hotspot": hotspot,
        "closure_best_operating_points": closure_best,
    }


def main() -> None:
    print("=" * 64)
    print("Gridlock precompute")
    print("=" * 64)
    service = get_service()
    master = build_master(service)

    areas = aggregate_areas(master)
    map_data = build_map(master)
    options = build_options(service, master, areas)
    metrics = build_metrics(master)

    _write("areas.json", areas)
    _write("map.json", map_data)
    _write("options.json", options)
    _write("metrics.json", metrics)
    print("[precompute] done.")


if __name__ == "__main__":
    main()
