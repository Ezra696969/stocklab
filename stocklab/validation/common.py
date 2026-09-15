"""Potongan yang dipakai bersama oleh semua pemeriksaan validasi."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from stocklab.backtest import run_backtest
from stocklab.backtest.metrics import (
    TRADING_DAYS,
    exposure,
    max_drawdown,
    sharpe_ratio,
    turnover_per_year,
)

__all__ = ["Experiment", "SignalBuilder", "evaluate", "rolling_windows"]

SignalBuilder = Callable[[pd.DataFrame, pd.Series, Mapping[str, Any]], pd.Series]
"""Menghasilkan bobot target dari fitur, label, dan parameter.

Sengaja berupa fungsi, bukan objek model: pemeriksaan stabilitas perlu
menjalankan ulang seluruh alur — termasuk walk-forward dan pelatihan ulang —
bukan sekadar memprediksi ulang dengan model yang sudah jadi.
"""


def evaluate(
    prices: pd.Series,
    signals: pd.Series,
    backtest_kwargs: Mapping[str, Any] | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """Jalankan backtest dan ambil metrik yang dipakai seluruh pemeriksaan.

    Sengaja ringkas: pemeriksaan stabilitas menjalankan ini ratusan kali, dan
    yang dibandingkan antar-run cuma segelintir angka.
    """
    result = run_backtest(prices, signals, **(backtest_kwargs or {}))
    benchmark_final = float(result.benchmark_equity.iloc[-1])
    final = float(result.equity.iloc[-1])

    return {
        "sharpe": sharpe_ratio(result.returns, periods_per_year),
        "benchmark_sharpe": sharpe_ratio(result.benchmark_returns, periods_per_year),
        "final_equity": final,
        "benchmark_equity": benchmark_final,
        "equity_ratio": final / benchmark_final if benchmark_final else float("nan"),
        "max_drawdown": max_drawdown(result.equity),
        "exposure": exposure(result.weights),
        "turnover_per_year": turnover_per_year(result.turnover, periods_per_year),
        "total_cost": result.total_cost,
    }


@dataclass(frozen=True)
class Experiment:
    """Satu alur lengkap yang bisa dijalankan ulang dengan variasi.

    Menyatukan data dan cara menghasilkan sinyal, sehingga pemeriksaan
    stabilitas bisa mengubah satu hal — parameter atau periode — sambil menjaga
    sisanya tetap sama.
    """

    prices: pd.Series
    features: pd.DataFrame
    target: pd.Series
    build_signals: SignalBuilder
    backtest_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features.index.equals(self.prices.index):
            raise ValueError("features dan prices harus berbagi index yang sama")
        if not self.target.index.equals(self.prices.index):
            raise ValueError("target dan prices harus berbagi index yang sama")

    def run(self, params: Mapping[str, Any] | None = None) -> dict[str, float]:
        """Jalankan sekali dan kembalikan metriknya."""
        signals = self.build_signals(self.features, self.target, dict(params or {}))
        return evaluate(self.prices, signals, self.backtest_kwargs)

    def signals(self, params: Mapping[str, Any] | None = None) -> pd.Series:
        return self.build_signals(self.features, self.target, dict(params or {}))

    def slice(self, start: Any, end: Any) -> Experiment:
        """Salinan yang dibatasi pada satu rentang tanggal."""
        window = slice(start, end)
        return Experiment(
            prices=self.prices.loc[window],
            features=self.features.loc[window],
            target=self.target.loc[window],
            build_signals=self.build_signals,
            backtest_kwargs=self.backtest_kwargs,
        )


def rolling_windows(
    index: pd.DatetimeIndex,
    window_years: int = 3,
    step_years: int = 1,
    min_coverage: float = 0.9,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Bagi rentang tanggal menjadi jendela tumpang tindih.

    Dipakai pemeriksaan stabilitas periode. Jendela yang tumpang tindih memang
    tidak saling bebas, jadi hasilnya dibaca sebagai gambaran konsistensi,
    bukan sebagai sekumpulan uji statistik yang berdiri sendiri.

    ``min_coverage`` menjaga jendela terakhir yang hampir penuh agar tidak
    terbuang. Data yang berakhir beberapa hari lebih cepat dari batas jendela
    tidak seharusnya menghapus periode terbaru sepenuhnya — dan periode terbaru
    justru yang paling sering berbeda dari sisanya, sehingga membuangnya diam-
    diam bisa membalik kesimpulan.
    """
    if len(index) == 0:
        return []
    if window_years < 1 or step_years < 1:
        raise ValueError("window_years dan step_years harus >= 1")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage harus di antara 0 dan 1")

    first, last = pd.Timestamp(index[0]), pd.Timestamp(index[-1])
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = first
    while True:
        end = start + pd.DateOffset(years=window_years) - pd.Timedelta(days=1)
        if end > last:
            break
        windows.append((start, end))
        start = start + pd.DateOffset(years=step_years)

    if start <= last:
        covered_years = (last - start).days / 365.25
        if covered_years >= window_years * min_coverage:
            windows.append((start, last))

    # Rentang yang lebih pendek dari satu jendela tetap layak dijalankan sekali.
    if not windows:
        windows.append((first, last))
    return windows
