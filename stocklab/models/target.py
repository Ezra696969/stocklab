"""Pembentukan label untuk pelatihan.

Label adalah satu-satunya tempat di seluruh paket ini yang secara sengaja
melihat ke depan: ``y[t]`` menjawab "apa yang terjadi setelah bar t". Itu sah
dan memang harus begitu — tanpanya tidak ada yang bisa dipelajari.

Yang berbahaya adalah akibatnya. Karena ``y[t]`` memuat informasi dari bar
``t + horizon``, baris latih terakhir selalu mengandung informasi yang, saat
backtest berjalan, belum terjadi. Itulah alasan walk-forward wajib memasang
jarak (embargo) minimal sebesar ``horizon`` antara akhir data latih dan awal
periode uji. Lihat :mod:`stocklab.backtest.walkforward`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["direction_target", "forward_return"]


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Return dari bar ``t`` ke bar ``t + horizon``.

    ``horizon`` bar terakhir bernilai NaN: masa depannya belum ada.
    """
    _check_horizon(horizon)
    return (close.shift(-horizon) / close - 1.0).rename(f"forward_return_{horizon}")


def direction_target(
    close: pd.Series, horizon: int = 1, threshold: float = 0.0
) -> pd.Series:
    """Label biner: 1 kalau return ke depan melampaui ``threshold``, 0 kalau tidak.

    ``threshold`` di atas nol membuat pergerakan yang terlalu kecil untuk
    menutup biaya transaksi tidak dihitung sebagai kenaikan. Model jadi tidak
    belajar mengejar gerakan yang tidak akan menguntungkan setelah biaya.
    """
    _check_horizon(horizon)
    future = forward_return(close, horizon)
    labels = (future > threshold).astype(float)
    labels[future.isna()] = np.nan
    return labels.rename(f"direction_{horizon}")


def _check_horizon(horizon: int) -> None:
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon harus bilangan bulat >= 1, dapat {horizon!r}")
