"""Uji rotasi: apakah waktu masuk-keluar model benar-benar membawa informasi.

Sebuah strategi bisa terlihat hebat hanya karena bentuk posisinya. Model yang
berdiam di kas tiga perempat waktu otomatis punya volatilitas rendah, dan
Sharpe-nya ikut naik tanpa satu pun prediksi yang benar. Membandingkannya
dengan buy & hold tidak menjawab itu, karena buy & hold selalu terisi penuh.

Uji ini membandingkan strategi dengan dirinya sendiri yang sudah diacak
waktunya. Deret sinyal digeser secara melingkar: jumlah bar terisi, panjang
tiap posisi, dan total turnover tetap sama, tapi keselarasan dengan pergerakan
harga hancur. Yang tersisa hanyalah bentuk posisi tanpa timing.

Kesamaannya berlaku hingga satu bar di batas putaran: ``execution_lag``
membuang sinyal terakhir, dan geseran menentukan sinyal mana yang terbuang.
Selisihnya satu bar dari beberapa ratus, jauh lebih kecil daripada efek yang
sedang diukur.

Kalau Sharpe asli tenggelam di dalam sebaran hasil geseran, keunggulannya
berasal dari bentuk itu, bukan dari kemampuan memprediksi. Kalau menonjol jauh
di atasnya, timing model memang membawa informasi.

Uji ini murah — tidak ada pelatihan ulang, hanya backtest berulang — dan
hampir tidak ada di perkakas backtesting ritel. Padahal inilah pemeriksaan yang
paling sering mementahkan strategi yang terlihat menjanjikan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from stocklab.backtest import run_backtest
from stocklab.backtest.metrics import TRADING_DAYS, sharpe_ratio

__all__ = ["RotationTest", "rotation_test"]


@dataclass(frozen=True)
class RotationTest:
    """Hasil uji rotasi."""

    observed: float
    """Nilai metrik pada sinyal yang sesungguhnya."""

    null: np.ndarray
    """Nilai metrik pada tiap geseran; sebaran di bawah hipotesis nihil."""

    p_value: float
    exhaustive: bool
    """``True`` kalau semua geseran yang mungkin diuji, bukan sampel acak."""

    metric_name: str = "sharpe"

    @property
    def n_rotations(self) -> int:
        return len(self.null)

    @property
    def null_median(self) -> float:
        return float(np.median(self.null))

    @property
    def null_max(self) -> float:
        return float(np.max(self.null))

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.null, q))

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value <= alpha

    def summary(self) -> dict[str, float]:
        return {
            "observed": self.observed,
            "null_median": self.null_median,
            "null_p90": self.percentile(90),
            "null_p99": self.percentile(99),
            "null_max": self.null_max,
            "p_value": self.p_value,
            "n_rotations": float(self.n_rotations),
        }


def _offsets(n: int, requested: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Pilih besar geseran; kalau semuanya muat, pakai semuanya.

    Geseran yang mungkin ada ``n - 1`` buah. Kalau permintaan mencakup semuanya,
    uji jadi eksak dan deterministik — tidak ada lagi ketergantungan pada seed,
    yang berarti hasilnya tidak bisa diulang-ulang sampai keluar angka yang
    menyenangkan.
    """
    available = n - 1
    if available <= 0:
        return np.empty(0, dtype=int), True
    if requested >= available:
        return np.arange(1, n, dtype=int), True
    return rng.choice(np.arange(1, n, dtype=int), size=requested, replace=False), False


def rotation_test(
    prices: pd.Series,
    signals: pd.Series,
    *,
    n_rotations: int = 500,
    metric: Callable[[pd.Series], float] | None = None,
    metric_name: str = "sharpe",
    backtest_kwargs: Mapping[str, Any] | None = None,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 0,
) -> RotationTest:
    """Uji sinyal terhadap versi dirinya sendiri yang digeser waktunya.

    Parameters
    ----------
    prices, signals:
        Sama seperti yang diberikan ke :func:`~stocklab.backtest.run_backtest`.
        Bar tanpa sinyal (NaN) tetap di tempatnya dan tidak ikut digeser: yang
        diacak adalah urutan keputusan, bukan kapan model mulai berpendapat.
    n_rotations:
        Banyak geseran yang dicoba. Kalau melebihi jumlah geseran yang mungkin,
        seluruhnya dipakai dan hasilnya menjadi eksak.
    metric:
        Fungsi yang menerima deret return strategi. Default Sharpe tahunan.
    backtest_kwargs:
        Diteruskan ke ``run_backtest``; biaya dan lag harus sama dengan run
        aslinya, atau perbandingannya tidak berarti.

    Notes
    -----
    p-value memakai koreksi ``(1 + k) / (1 + n)``. Tanpa itu, hasil yang tidak
    pernah terlampaui akan dilaporkan sebagai ``p = 0``, seolah-olah
    kemungkinannya nol — padahal yang sebenarnya diketahui hanyalah bahwa nilai
    itu lebih kecil dari satu per jumlah geseran.
    """
    if not isinstance(signals, pd.Series):
        raise TypeError("signals harus pd.Series agar sejajar dengan prices")
    if not signals.index.equals(prices.index):
        raise ValueError("index signals tidak sama dengan index prices")
    if n_rotations < 1:
        raise ValueError("n_rotations harus >= 1")

    kwargs = dict(backtest_kwargs or {})
    scorer = metric or (lambda returns: sharpe_ratio(returns, periods_per_year))

    observed = float(scorer(run_backtest(prices, signals, **kwargs).returns))

    values = signals.to_numpy(dtype=float)
    present = ~np.isnan(values)
    n_present = int(present.sum())
    if n_present < 3:
        raise ValueError(
            f"butuh minimal 3 bar bersinyal untuk diputar, dapat {n_present}"
        )

    shifts, exhaustive = _offsets(n_present, n_rotations, np.random.default_rng(seed))
    active = values[present]

    null = np.empty(len(shifts), dtype=float)
    for i, shift in enumerate(shifts):
        rotated = values.copy()
        rotated[present] = np.roll(active, int(shift))
        candidate = pd.Series(rotated, index=signals.index)
        null[i] = float(scorer(run_backtest(prices, candidate, **kwargs).returns))

    exceeded = int(np.sum(null >= observed))
    p_value = (1.0 + exceeded) / (1.0 + len(null))

    return RotationTest(
        observed=observed,
        null=null,
        p_value=p_value,
        exhaustive=exhaustive,
        metric_name=metric_name,
    )
