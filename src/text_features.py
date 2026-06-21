"""Multilingual text features for the free-text ``description`` field.

The descriptions mix English, transliterated Kannada and native Kannada script
and often state the traffic impact directly ("road closed", "slow moment",
"traffic normal"). We encode them with a multilingual sentence-transformer
(LaBSE-style) that understands Kannada, cache the raw embeddings to disk, and
let the modelling layer reduce them with a train-fitted PCA.

A TF-IDF + SVD fallback is provided for CPU-only / offline environments
(``GRIDLOCK_NO_TRANSFORMER=1``) so the pipeline always runs.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from . import config as C

_ANON_TOKEN_RE = re.compile(r"\[(?:PERSON|LOCATION|PHONE|EMAIL|ID)\]")
_WS_RE = re.compile(r"\s+")


def clean_text(series: pd.Series) -> pd.Series:
    """Strip anonymisation tokens and collapse whitespace."""
    s = series.fillna("").astype(str)
    s = s.str.replace(_ANON_TOKEN_RE, " ", regex=True)
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def _cache_key(texts: list[str]) -> str:
    h = hashlib.md5()
    h.update(C.EMBED_MODEL_NAME.encode())
    h.update(str(len(texts)).encode())
    h.update("".join(texts[:50]).encode("utf-8", "ignore"))
    return h.hexdigest()[:12]


def _transformer_embeddings(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(C.EMBED_MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def _tfidf_embeddings(texts: list[str], n_components: int = 64) -> np.ndarray:
    """Offline fallback: char+word TF-IDF reduced with truncated SVD.

    Character n-grams make this robust to Kannada script and transliteration.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=20000
    )
    X = vec.fit_transform(texts)
    k = min(n_components, X.shape[1] - 1, max(2, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=C.RANDOM_STATE)
    return svd.fit_transform(X).astype(np.float32)


def compute_embeddings(df: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
    """Return a (n_rows, dim) raw embedding matrix, cached to disk by content."""
    texts = clean_text(df[C.TEXT_COLUMN]).tolist()
    key = _cache_key(texts)
    cache_path = C.PROCESSED_DIR / f"text_embeddings_{key}.npy"

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    if C.USE_TRANSFORMER:
        try:
            emb = _transformer_embeddings(texts)
        except Exception as exc:  # pragma: no cover - network/model failure
            print(f"[text_features] transformer unavailable ({exc}); using TF-IDF fallback")
            emb = _tfidf_embeddings(texts)
    else:
        emb = _tfidf_embeddings(texts)

    if use_cache:
        np.save(cache_path, emb)
        np.save(C.EMBED_CACHE, emb)
    return emb


if __name__ == "__main__":  # pragma: no cover
    d = pd.read_parquet(C.FEATURES_PARQUET if C.FEATURES_PARQUET.exists() else C.CLEAN_PARQUET)
    e = compute_embeddings(d)
    print("embeddings shape:", e.shape, "dtype:", e.dtype)
