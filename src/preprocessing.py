"""Feature-matrix assembly with strictly train-fitted transforms.

The :class:`Preprocessor` learns everything it needs (categorical vocabularies,
embedding PCA, numeric medians) from the *training* rows only, then applies the
same transform to validation / test / future data. This is what keeps the
evaluation honest - no test statistic ever touches the fitted artefacts.

Categorical columns are emitted as pandas ``category`` dtype so gradient-boosted
trees can split on them natively (unseen test categories become NaN, which the
trees route automatically). Text embeddings are reduced with PCA fitted on the
training split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Preprocessor:
    target: str
    use_embeddings: bool = True
    n_emb_components: int | None = None  # None -> C.EMBED_PCA_COMPONENTS
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    _cat_categories: dict[str, np.ndarray] = field(default_factory=dict)
    _numeric_median: dict[str, float] = field(default_factory=dict)
    _pca = None
    _emb_mean = None
    feature_names_: list[str] = field(default_factory=list)

    def _select_columns(self, df: pd.DataFrame):
        exclude = set()
        if self.target == C.TARGET_PRIORITY:
            exclude.update(C.PRIORITY_EXCLUDE_FEATURES)
        num = [c for c in C.NUMERIC_FEATURES if c in df.columns and c not in exclude]
        cat = [c for c in C.CATEGORICAL_FEATURES if c in df.columns and c not in exclude]
        return num, cat

    def fit(self, df_train: pd.DataFrame, emb_train: np.ndarray | None = None):
        self.numeric_cols, self.categorical_cols = self._select_columns(df_train)

        for c in self.numeric_cols:
            self._numeric_median[c] = float(
                pd.to_numeric(df_train[c], errors="coerce").median()
            )
        for c in self.categorical_cols:
            cats = (
                df_train[c].astype("object").where(df_train[c].notna(), np.nan)
                .dropna().astype(str).unique()
            )
            self._cat_categories[c] = np.sort(cats)

        if self.use_embeddings and emb_train is not None:
            from sklearn.decomposition import PCA

            target_comp = self.n_emb_components or C.EMBED_PCA_COMPONENTS
            n_comp = min(target_comp, emb_train.shape[1], emb_train.shape[0] - 1)
            self._emb_mean = emb_train.mean(axis=0)
            self._pca = PCA(n_components=n_comp, random_state=C.RANDOM_STATE)
            self._pca.fit(emb_train - self._emb_mean)

        self.feature_names_ = list(self.numeric_cols) + list(self.categorical_cols)
        if self._pca is not None:
            self.feature_names_ += [f"emb_{i}" for i in range(self._pca.n_components_)]
        return self

    def transform(self, df: pd.DataFrame, emb: np.ndarray | None = None) -> pd.DataFrame:
        out = {}
        for c in self.numeric_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            out[c] = col.fillna(self._numeric_median[c]).astype(np.float32)
        num_df = pd.DataFrame(out, index=df.index)

        cat_df = pd.DataFrame(index=df.index)
        for c in self.categorical_cols:
            vals = df[c].astype("object").where(df[c].notna(), np.nan).astype("string")
            cat_df[c] = pd.Categorical(vals, categories=self._cat_categories[c])

        parts = [num_df, cat_df]
        if self._pca is not None and emb is not None:
            reduced = self._pca.transform(emb - self._emb_mean).astype(np.float32)
            emb_df = pd.DataFrame(
                reduced, columns=[f"emb_{i}" for i in range(reduced.shape[1])],
                index=df.index,
            )
            parts.append(emb_df)

        X = pd.concat(parts, axis=1)
        return X[self.feature_names_]

    @property
    def categorical_feature_names(self) -> list[str]:
        return list(self.categorical_cols)
