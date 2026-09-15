"""Fixture bersama untuk pengujian."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 400, seed: int = 21) -> pd.DataFrame:
    """OHLCV sintetis yang deterministik dan konsisten secara internal.

    High benar-benar tertinggi dan low benar-benar terendah, karena beberapa
    indikator akan menghasilkan angka mustahil kalau syarat itu dilanggar.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
    spread = np.abs(rng.normal(0.0, 0.006, n)) * close

    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000_000, 9_000_000, n).astype(float)

    index = pd.date_range("2023-01-02", periods=n, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()
