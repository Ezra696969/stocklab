"""Stabilitas hasil terhadap pilihan parameter dan pilihan periode.

Dua pertanyaan berbeda yang gampang tertukar.

**Stabilitas parameter** menjawab: apakah hasil ini bertahan kalau slider
digeser sedikit? Puncak sempit yang runtuh begitu satu parameter berubah adalah
tanda model menghafal, bukan menemukan. Sebaliknya, hasil yang konsisten
sepanjang grid biasanya menandakan efeknya nyata.

**Stabilitas periode** menjawab: apakah hasil ini muncul di rentang waktu lain,
atau hanya pada satu potongan sejarah? Ini yang paling sering mementahkan
strategi yang lolos semua pemeriksaan lain — dan yang paling jarang diperiksa,
karena hasilnya paling sering mengecewakan.

Keduanya melatih ulang model dari nol untuk tiap variasi, sehingga jauh lebih
mahal daripada uji rotasi. Itu harga yang wajar: keduanya menjawab pertanyaan
yang tidak bisa dijawab dengan menghitung ulang metrik dari sinyal yang sama.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stocklab.validation.common import Experiment, rolling_windows

__all__ = [
    "StabilityReport",
    "parameter_stability",
    "period_stability",
]


@dataclass(frozen=True)
class StabilityReport:
    """Sekumpulan run yang hanya berbeda pada satu hal."""

    table: pd.DataFrame
    """Satu baris per variasi, berisi metrik dan variasi yang dipakai."""

    varied: str
    """Apa yang diubah: ``"parameters"`` atau ``"period"``."""

    @property
    def n_runs(self) -> int:
        return len(self.table)

    @property
    def n_failed(self) -> int:
        """Run yang tidak menghasilkan Sharpe, biasanya karena data tak cukup."""
        return int(self.table["sharpe"].isna().sum())

    @property
    def win_rate(self) -> float:
        """Bagian run yang mengungguli pasif secara nominal."""
        ratios = self.table["equity_ratio"].dropna()
        return float((ratios > 1.0).mean()) if len(ratios) else float("nan")

    @property
    def sharpe_win_rate(self) -> float:
        """Bagian run yang mengungguli pasif secara risk-adjusted.

        Dipisah dari :attr:`win_rate` karena keduanya sering tidak sejalan:
        strategi yang banyak berdiam di kas bisa konsisten unggul Sharpe sambil
        konsisten kalah nominal.
        """
        usable = self.table[["sharpe", "benchmark_sharpe"]].dropna()
        if usable.empty:
            return float("nan")
        return float((usable["sharpe"] > usable["benchmark_sharpe"]).mean())

    def sharpe_spread(self) -> dict[str, float]:
        values = self.table["sharpe"].dropna()
        if values.empty:
            return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
        return {
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
        }

    def worst(self) -> pd.Series | None:
        """Run terburuk. Inilah yang seharusnya dilihat user lebih dulu."""
        values = self.table["sharpe"].dropna()
        return None if values.empty else self.table.loc[values.idxmin()]

    def sharpes(self) -> np.ndarray:
        """Sharpe tiap run, untuk diumpankan ke deflated Sharpe.

        Sebaran ini jauh lebih informatif daripada sekadar jumlah percobaan:
        pencarian yang hasilnya berserakan lebar gampang melahirkan pemenang
        kebetulan.
        """
        return self.table["sharpe"].dropna().to_numpy()

    def summary(self) -> dict[str, float]:
        spread = self.sharpe_spread()
        return {
            "n_runs": float(self.n_runs),
            "n_failed": float(self.n_failed),
            "win_rate": self.win_rate,
            "sharpe_win_rate": self.sharpe_win_rate,
            "sharpe_min": spread["min"],
            "sharpe_median": spread["median"],
            "sharpe_max": spread["max"],
        }


def _safe_run(experiment: Experiment, params: Mapping[str, Any]) -> dict[str, float]:
    """Jalankan satu variasi; kegagalan dicatat, bukan menghentikan sweep.

    Sebagian kombinasi memang tidak layak — jendela yang terlalu pendek untuk
    walk-forward, atau label yang seragam. Menghentikan seluruh sweep karena
    satu kombinasi buruk akan membuang hasil yang sudah terkumpul.
    """
    try:
        return experiment.run(params)
    except (ValueError, RuntimeError) as error:
        return {"sharpe": float("nan"), "error": str(error)}


def parameter_stability(
    experiment: Experiment, grid: Mapping[str, Sequence[Any]]
) -> StabilityReport:
    """Jalankan seluruh kombinasi di ``grid`` dan bandingkan hasilnya.

    Parameters
    ----------
    grid:
        Nama parameter dan nilai yang ingin dicoba. Seluruh kombinasi dijalankan
        (product), jadi grid besar berlipat dengan cepat.

    Notes
    -----
    Jumlah run yang dihasilkan di sini adalah jumlah percobaan yang harus
    dilaporkan ke :func:`~stocklab.backtest.metrics.deflated_sharpe_ratio`.
    Melihat seluruh grid lalu melaporkan yang terbaik, tanpa menyebut sisanya,
    adalah bentuk paling umum dari melebih-lebihkan hasil backtest.
    """
    if not grid:
        raise ValueError("grid kosong; tidak ada yang bisa dibandingkan")

    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[name] for name in names)):
        params = dict(zip(names, values))
        rows.append({**params, **_safe_run(experiment, params)})

    return StabilityReport(table=pd.DataFrame(rows), varied="parameters")


def period_stability(
    experiment: Experiment,
    *,
    window_years: int = 3,
    step_years: int = 1,
    windows: Sequence[tuple[Any, Any]] | None = None,
    params: Mapping[str, Any] | None = None,
) -> StabilityReport:
    """Jalankan alur yang sama pada beberapa rentang waktu.

    Parameters
    ----------
    window_years, step_years:
        Panjang dan jarak antar jendela, kalau ``windows`` tidak diberikan.
    windows:
        Rentang eksplisit, menimpa dua parameter di atas.
    params:
        Parameter model, sama untuk semua jendela — yang diuji di sini adalah
        periodenya, bukan parameternya.
    """
    index = experiment.prices.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("stabilitas periode butuh index bertanggal")

    ranges = (
        list(windows)
        if windows is not None
        else rolling_windows(index, window_years, step_years)
    )
    if not ranges:
        raise ValueError("tidak ada jendela yang bisa dibentuk dari rentang data")

    rows = []
    for start, end in ranges:
        sliced = experiment.slice(start, end)
        metrics = (
            _safe_run(sliced, dict(params or {}))
            if len(sliced.prices) > 1
            else {"sharpe": float("nan"), "error": "jendela terlalu pendek"}
        )
        rows.append(
            {
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(end),
                "n_bars": len(sliced.prices),
                **metrics,
            }
        )

    return StabilityReport(table=pd.DataFrame(rows), varied="period")
