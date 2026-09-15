"""Sensitivitas terhadap biaya transaksi.

Strategi dengan turnover tinggi hidup atau mati di asumsi biaya, dan asumsi itu
biasanya diketik sekali lalu dilupakan. Satu angka biaya menghasilkan satu
equity curve yang terlihat meyakinkan; yang perlu diketahui user adalah pada
biaya berapa keunggulan itu habis, dan apakah angka tersebut masih di atas
biaya yang benar-benar mereka bayar.

Titik impas dilaporkan sebagai bps per satu unit turnover — yaitu biaya satu
sisi transaksi, karena masuk penuh dari kas bernilai turnover 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stocklab.validation.common import evaluate

__all__ = ["CostSensitivity", "cost_sensitivity"]

DEFAULT_LEVELS: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0)


@dataclass(frozen=True)
class CostSensitivity:
    """Performa pada beberapa tingkat biaya."""

    table: pd.DataFrame
    """Terindeks ``cost_bps``, berisi sharpe, equity, dan rasio terhadap pasif."""

    breakeven_bps: float | None
    """Biaya saat strategi berhenti mengungguli pasif secara nominal.

    ``None`` kalau strategi tidak pernah unggul, atau unggul di seluruh tingkat
    yang diuji. Bedanya terbaca dari kolom ``equity_ratio``.
    """

    turnover_per_year: float

    @property
    def beats_at_zero_cost(self) -> bool:
        return bool(self.table["equity_ratio"].iloc[0] > 1.0)

    def survives(self, cost_bps: float) -> bool:
        """Apakah strategi masih unggul pada tingkat biaya tertentu."""
        if self.breakeven_bps is None:
            return self.beats_at_zero_cost
        return cost_bps < self.breakeven_bps

    def summary(self) -> dict[str, float | None]:
        return {
            "breakeven_bps": self.breakeven_bps,
            "turnover_per_year": self.turnover_per_year,
            "best_sharpe": float(self.table["sharpe"].max()),
            "worst_sharpe": float(self.table["sharpe"].min()),
        }


def _interpolate_crossing(levels: np.ndarray, ratios: np.ndarray) -> float | None:
    """Cari biaya saat ``equity_ratio`` memotong 1.0, secara linier."""
    above = ratios > 1.0
    if not above.any():
        return None
    if above.all():
        return None

    # Titik impas pertama: peralihan dari unggul ke tidak.
    for i in range(len(ratios) - 1):
        if above[i] and not above[i + 1]:
            x0, x1 = levels[i], levels[i + 1]
            y0, y1 = ratios[i], ratios[i + 1]
            if y0 == y1:
                return float(x1)
            return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))
    return None


def cost_sensitivity(
    prices: pd.Series,
    signals: pd.Series,
    *,
    levels: Sequence[float] = DEFAULT_LEVELS,
    backtest_kwargs: Mapping[str, Any] | None = None,
) -> CostSensitivity:
    """Jalankan ulang backtest pada beberapa tingkat biaya.

    Parameters
    ----------
    levels:
        Biaya dalam bps per unit turnover, harus menaik. ``cost_bps`` dan
        ``slippage_bps`` di ``backtest_kwargs`` diabaikan: nilai di sinilah
        yang dipakai, supaya tidak ada biaya yang terhitung dua kali.
    """
    ordered = np.asarray(levels, dtype=float)
    if len(ordered) < 2:
        raise ValueError("butuh minimal dua tingkat biaya untuk dibandingkan")
    if np.any(np.diff(ordered) <= 0):
        raise ValueError("levels harus menaik dan tanpa duplikat")
    if np.any(ordered < 0):
        raise ValueError("biaya tidak boleh negatif")

    kwargs = dict(backtest_kwargs or {})
    kwargs.pop("cost_bps", None)
    kwargs.pop("slippage_bps", None)

    rows = []
    for level in ordered:
        metrics = evaluate(
            prices, signals, {**kwargs, "cost_bps": float(level), "slippage_bps": 0.0}
        )
        rows.append(metrics)

    table = pd.DataFrame(rows, index=pd.Index(ordered, name="cost_bps"))
    breakeven = _interpolate_crossing(ordered, table["equity_ratio"].to_numpy())

    return CostSensitivity(
        table=table,
        breakeven_bps=breakeven,
        turnover_per_year=float(table["turnover_per_year"].iloc[0]),
    )
