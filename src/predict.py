"""Inference: load trained artifacts and produce forecasts + recommendations.

``predict_events(df_raw)`` accepts raw event rows (same schema as the source
CSV), runs the exact cleaning / feature / embedding pipeline used in training,
and returns one :class:`~src.recommend.Recommendation` per row plus the raw
model outputs. This is the deployable entry point a control-room tool would call.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .cleaning import clean
from .feature_engineering import build_features, load_history
from .recommend import recommend
from .targets import build_targets
from .text_features import compute_embeddings


class GridlockPredictor:
    def __init__(self):
        self.closure = joblib.load(C.MODELS_DIR / "closure_model.joblib")
        self.priority = joblib.load(C.MODELS_DIR / "priority_model.joblib")
        self.duration = joblib.load(C.MODELS_DIR / "duration_model.joblib")
        self.prep_full = joblib.load(C.MODELS_DIR / "preprocessor_full.joblib")
        self.prep_priority = joblib.load(C.MODELS_DIR / "preprocessor_priority.joblib")
        self.prep_duration = joblib.load(C.MODELS_DIR / "preprocessor_duration.joblib")
        self.history = load_history()

    def _prepare(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df = clean(df_raw, save=False)
        df = build_targets(df, save=False)        # harmless if labels absent
        # training=False loads the persisted geo KMeans; history reproduces the
        # past-only causal target-rate features the models trained on.
        df = build_features(df, save=False, training=False, history=self.history)
        emb = compute_embeddings(df, use_cache=False)
        return df.reset_index(drop=True), emb

    def predict_frame(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df, emb = self._prepare(df_raw)

        Xc = self.prep_full.transform(df, emb)
        Xp = self.prep_priority.transform(df, emb)
        Xd = self.prep_duration.transform(df, emb)

        closure_prob = self.closure.predict_proba(Xc)
        closure_pred = (closure_prob >= self.closure.threshold).astype(int)
        priority_prob = self.priority.predict_proba(Xp)
        point = self.duration.predict_minutes(Xd)
        quant = self.duration.predict_quantiles(Xd)

        recs = []
        for i in range(len(df)):
            rec = recommend(
                closure_prob=float(closure_prob[i]),
                closure_pred=int(closure_pred[i]),
                priority_prob=float(priority_prob[i]),
                duration_min=float(point[i]),
                duration_low=float(quant[0.1][i]),
                duration_high=float(quant[0.9][i]),
                closure_threshold=float(self.closure.threshold),
            )
            recs.append(rec.to_dict())

        out = pd.DataFrame(recs)
        if "id" in df.columns:
            out.insert(0, "event_id", df["id"].values)
        for col in ("event_type", "event_cause", "address"):
            if col in df.columns:
                out[col] = df[col].values
        return out


def predict_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    return GridlockPredictor().predict_frame(df_raw)


if __name__ == "__main__":  # pragma: no cover
    from .data_loading import load_raw

    raw = load_raw()
    sample = raw.tail(8).copy()
    result = predict_events(sample)
    cols = ["closure_probability", "closure_expected", "high_priority_probability",
            "expected_duration_min", "manpower_tier", "officers_suggested"]
    print(result[[c for c in cols if c in result.columns]].to_string())
