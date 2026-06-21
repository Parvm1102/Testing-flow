"""Raw data loading utilities."""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw Astram event CSV with every column read as a string.

    Reading as string first lets us normalise the many textual NULL tokens
    (``NULL``, ``None``, ``[]`` ...) consistently before type conversion.
    """
    path = path or C.RAW_CSV
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    df.columns = [c.strip() for c in df.columns]
    return df


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    d = load_raw()
    print(d.shape)
    print(d.columns.tolist())
