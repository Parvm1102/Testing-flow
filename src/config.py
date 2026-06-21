"""Central configuration: paths, column groupings, and modeling constants.

The column groupings here are the single most important guard against data
leakage. Any column produced *after* an event is reported (resolution times,
status, post-hoc edits, assigned officers) must never become a model feature.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = RAW_DIR / "astram_events.csv"
CLEAN_PARQUET = PROCESSED_DIR / "events_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
EMBED_CACHE = PROCESSED_DIR / "text_embeddings.npy"
# Persisted artefacts that make the causal/spatial features reproducible at
# inference time (a new event must see the SAME geo bins and the accumulated
# historical target rates the model was trained on).
HISTORY_PARQUET = PROCESSED_DIR / "history.parquet"
GEO_KMEANS_PATH = MODELS_DIR / "geo_kmeans.joblib"

for _d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Raw column groupings
# --------------------------------------------------------------------------- #
# Identifiers / opaque tokens -> dropped from features entirely.
ID_COLUMNS = [
    "id",
    "kgid",
    "veh_no",
    "client_id",
    "created_by_id",
    "last_modified_by_id",
    "assigned_to_police_id",  # post-assignment -> would leak the resourcing label
    "citizen_accident_id",
    "closed_by_id",
    "resolved_by_id",
    "map_file",
    "meta_data",
    "gba_identifier",
]

# Columns only known AFTER the event unfolds -> never features. Some are used
# strictly to construct duration / status labels in targets.py.
LEAKAGE_COLUMNS = [
    "status",
    "modified_datetime",
    "end_datetime",        # used to derive duration label only
    "closed_datetime",     # used to derive duration label only
    "resolved_datetime",   # used to derive duration label only
    "resolved_at_address",
    "resolved_at_latitude",
    "resolved_at_longitude",
    "comment",             # frequently appended after resolution
    # The end-point / diversion-route geometry is only populated when a closure
    # or diversion SEGMENT is drawn -> it encodes the closure decision itself
    # (has_end_point alone gives AP~0.98 on closure). Treat as leakage so the
    # model must forecast from genuine report-time signal instead.
    "endlatitude",
    "endlongitude",
    "end_address",
    "route_path",
]

# The three prediction targets (also excluded from the feature matrix).
TARGET_CLOSURE = "y_road_closure"     # binary  (barricading / diversion need)
TARGET_PRIORITY = "y_high_priority"   # binary  (manpower tier)
TARGET_DURATION = "y_duration_min"    # numeric (impact duration, minutes)
TARGET_DURATION_LOG = "y_duration_log"  # log1p transform actually modeled
TARGET_COLUMNS_RAW = ["requires_road_closure", "priority"]

# Timestamp columns parsed to tz-aware datetimes.
DATETIME_COLUMNS = [
    "start_datetime",
    "end_datetime",
    "modified_datetime",
    "created_date",
    "closed_datetime",
    "resolved_datetime",
]

# --------------------------------------------------------------------------- #
# Feature columns (available at event-creation time)
# --------------------------------------------------------------------------- #
CATEGORICAL_FEATURES = [
    "event_type",
    "event_cause",
    "event_family",      # engineered grouping of event_cause
    "veh_type",
    "direction",
    "corridor",
    "zone",
    "junction",
    "police_station",
    "reason_breakdown",
    "cargo_material",
    "authenticated",
    "pincode",           # engineered from address
    "geo_cluster",       # engineered unsupervised spatial bin
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "age_of_truck",
    # temporal
    "start_hour",
    "start_dow",
    "start_month",
    "start_weekofyear",
    "start_day",
    "is_weekend",
    "is_morning_peak",
    "is_evening_peak",
    "is_night",
    "lead_time_hours",
    "has_advance_notice",
    # cyclical encodings
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # spatial (report-time location only; end-point/route geometry is leakage)
    "hist_hotspot_count",
    "loc_event_density",
    # recurrence
    "same_loc_cause_hist",
    "same_day_loc_reports",
    # causal target-rate encodings (past-only smoothed closure rate per group;
    # adapts to drift and gives the rare closure target far more signal than the
    # static category alone). See feature_engineering._causal_target_features.
    "clo_rate_family",
    "clo_rate_cause",
    "clo_rate_corridor",
    "clo_rate_police",
    "clo_rate_junction",
    "clo_rate_geocluster",
    "clo_rate_zone",
    "clo_rate_pincode",
    # ambient duration level: rolling mean log-duration of recently-resolved
    # events (tracks the heavy non-stationarity in clearance times).
    "dur_recent_level",
    # text-derived scalar features
    "desc_len",
    "desc_word_count",
    "desc_is_kannada",
    "desc_has_text",
    "lex_closed",
    "lex_slow",
    "lex_normal",
    "lex_diversion",
    "lex_blocked",
    "lex_heavy",
    "lex_festival",
    # missingness flags
    "veh_type_missing",
    "desc_missing",
]

# Free-text column for embeddings / lexicon.
TEXT_COLUMN = "description"

# --------------------------------------------------------------------------- #
# event_cause -> event_family mapping
# Keys are lowercased/stripped event_cause values (the raw data mixes case,
# e.g. "Debris"/"debris", and has 17 distinct causes).
# --------------------------------------------------------------------------- #
EVENT_FAMILY_MAP = {
    # event-driven / planned gatherings (the core of the problem statement)
    "public_event": "gathering",
    "procession": "gathering",
    "protest": "gathering",
    "vip_movement": "vip",
    "construction": "construction",
    # unplanned incidents
    "vehicle_breakdown": "breakdown",
    "accident": "accident",
    "tree_fall": "obstruction",
    "water_logging": "obstruction",
    "debris": "obstruction",
    "fog / low visibility": "obstruction",
    "pot_holes": "road_condition",
    "road_conditions": "road_condition",
    "congestion": "congestion",
    "test_demo": "other",
    "others": "other",
}
EVENT_FAMILY_DEFAULT = "other"

# Features excluded from the PRIORITY model only: `corridor` is ~99.8%
# deterministic of `priority` (High == named corridor, Low == Non-corridor),
# so using it makes the task trivially leaky. We force the model to infer
# priority from geography / cause / text / time instead.
PRIORITY_EXCLUDE_FEATURES = ["corridor"]

# --------------------------------------------------------------------------- #
# Multilingual impact lexicon (English + transliterated + Kannada)
# Used both as interpretable features and as a fallback when transformer
# embeddings are unavailable.
# --------------------------------------------------------------------------- #
LEXICON = {
    "lex_closed": ["closed", "close", "block", "blocked", "shut", "ಮುಚ್ಚ", "ಬಂದ್", "ಕ್ಲೋಸ್"],
    "lex_slow": ["slow", "slowly", "moment", "movement", "nidhana", "ನಿಧಾನ", "ಸ್ಲೋ", "slow moment"],
    "lex_normal": ["normal", "no problem", "no traffic", "clear", "ಸಮಸ್ಯ ಇಲ್ಲ", "ಯಾವುದೇ ತೊಂದರೆ"],
    "lex_diversion": ["diversion", "divert", "u turn", "uturn", "alternate", "ಡೈವರ್ಶನ್", "ಬದಲಾವಣೆ"],
    "lex_blocked": ["jam", "gridlock", "heavy traffic", "congestion", "ಟ್ರಾಫಿಕ್ ಜಾಮ್", "ದಟ್ಟಣೆ"],
    "lex_heavy": ["lorry", "truck", "bus", "heavy", "container", "ಲಾರಿ", "ಬಸ್", "ಟ್ರಕ್"],
    "lex_festival": ["festival", "utsava", "uthsava", "palakki", "pallakki", "procession", "rally", "ಉತ್ಸವ", "ಪಲ್ಲಕ್ಕಿ", "ಹಬ್ಬ"],
}

# --------------------------------------------------------------------------- #
# Modeling constants
# --------------------------------------------------------------------------- #
# Fraction of the (time-ordered) data held out as the final temporal test set.
TEST_FRACTION = 0.20
N_FOLDS = 5
# Number of unsupervised spatial bins (KMeans on report-time coordinates).
GEO_CLUSTERS = 40
# Duration label handling
DURATION_MIN_MINUTES = 1.0      # drop non-positive / sub-minute durations
DURATION_WINSOR_UPPER_Q = 0.99  # cap extreme long-tail durations

# Causal target-rate encoding. Each (group key -> column) pair becomes a
# past-only smoothed mean of the closure label within that group. Smoothing
# shrinks low-count groups toward the running global rate (empirical-Bayes).
CAUSAL_TARGET_SMOOTHING = 25
CLOSURE_RATE_KEYS = {
    "event_family": "clo_rate_family",
    "event_cause": "clo_rate_cause",
    "corridor": "clo_rate_corridor",
    "police_station": "clo_rate_police",
    "junction": "clo_rate_junction",
    "geo_cluster": "clo_rate_geocluster",
    "zone": "clo_rate_zone",
    "pincode": "clo_rate_pincode",
}
# Window (in recently-resolved events) for the ambient duration-level feature.
DUR_RECENT_WINDOW = 200

# Transformer embedding model (multilingual, supports Kannada). Set the env var
# GRIDLOCK_NO_TRANSFORMER=1 to force the lightweight TF-IDF fallback.
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Default PCA width for the 384-dim embeddings. Kept modest (32) for the
# duration / priority models: duration has only ~2.5k valid rows, so wide
# embeddings overfit (validated: R2_log 0.139@32 -> 0.10@128). The closure
# model, trained on all rows, benefits from a wider projection and overrides
# this with CLOSURE_EMB_PCA_COMPONENTS (validated: AP 0.314@32 -> 0.335@96).
EMBED_PCA_COMPONENTS = 32
CLOSURE_EMB_PCA_COMPONENTS = 96
USE_TRANSFORMER = os.environ.get("GRIDLOCK_NO_TRANSFORMER", "0") != "1"

# Optuna
OPTUNA_TRIALS = int(os.environ.get("GRIDLOCK_OPTUNA_TRIALS", "40"))
OPTUNA_TIMEOUT = int(os.environ.get("GRIDLOCK_OPTUNA_TIMEOUT", "600"))  # seconds/target

NULL_TOKENS = {"NULL", "null", "None", "none", "", "nan", "NaN", "[]"}
