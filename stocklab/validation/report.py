"""Gabungan seluruh pemeriksaan menjadi satu laporan.

Empat pemeriksaan menjawab empat pertanyaan berbeda, dan sebuah strategi bisa
lolos tiga lalu gugur di yang keempat. Laporan ini menyatukannya, lalu
menerjemahkan hasilnya menjadi daftar kekhawatiran yang bisa dibaca —
karena angka p-value dan deflated Sharpe tidak berarti apa-apa bagi user yang
baru pertama kali melihatnya.

Yang sengaja tidak dilakukan di sini: memberi vonis lulus atau gagal. Ambang
apa pun yang dipilih akan sewenang-wenang, dan label "LULUS" pada sebuah
strategi jauh lebih berbahaya daripada tidak ada label sama sekali. Yang
diberikan adalah daftar hal yang perlu diketahui user, dengan angkanya.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stocklab.backtest.metrics import deflated_sharpe_ratio
from stocklab.validation.common import Experiment, evaluate, tradable_slice
from stocklab.validation.costs import CostSensitivity, cost_sensitivity
from stocklab.validation.rotation import RotationTest, rotation_test
from stocklab.validation.stability import (
    StabilityReport,
    parameter_stability,
    period_stability,
)

__all__ = ["ValidationReport", "validate"]

#: Ambang yang dipakai untuk memunculkan kekhawatiran. Longgar dengan sengaja:
#: tujuannya menarik perhatian, bukan menjatuhkan vonis.
ALPHA = 0.05
MIN_WIN_RATE = 0.5
MIN_DEFLATED_SHARPE = 0.95
COST_SAFETY_FACTOR = 2.0


@dataclass(frozen=True)
class ValidationReport:
    """Hasil seluruh pemeriksaan untuk satu strategi."""

    baseline: dict[str, float]
    """Metrik run utama, sebagai pembanding semua pemeriksaan lain."""

    rotation: RotationTest | None = None
    costs: CostSensitivity | None = None
    parameters: StabilityReport | None = None
    periods: StabilityReport | None = None
    deflated_sharpe: float = float("nan")
    assumed_cost_bps: float = 0.0

    @property
    def concerns(self) -> list[str]:
        """Hal yang perlu diketahui user sebelum mempercayai hasil ini.

        Daftar kosong bukan berarti strategi ini bagus. Artinya hanya:
        pemeriksaan yang dijalankan tidak menemukan masalah yang dicarinya.
        """
        found: list[str] = []

        if self.rotation is not None and not self.rotation.is_significant(ALPHA):
            found.append(
                f"Timing model tidak terbukti membawa informasi "
                f"(p = {self.rotation.p_value:.3f}). Sinyal acak dengan pola "
                f"posisi yang sama menghasilkan performa serupa, jadi "
                f"keunggulannya kemungkinan berasal dari bentuk posisi, bukan "
                f"dari prediksi."
            )

        if self.costs is not None:
            breakeven = self.costs.breakeven_bps
            if breakeven is None and not self.costs.beats_at_zero_cost:
                found.append(
                    "Strategi tidak mengungguli pasif bahkan tanpa biaya sama sekali."
                )
            elif breakeven is not None:
                if breakeven <= self.assumed_cost_bps:
                    found.append(
                        f"Titik impas biaya ({breakeven:.0f} bps) berada di bawah "
                        f"biaya yang diasumsikan ({self.assumed_cost_bps:.0f} bps)."
                    )
                elif breakeven < self.assumed_cost_bps * COST_SAFETY_FACTOR:
                    found.append(
                        f"Titik impas biaya cuma {breakeven:.0f} bps, terlalu dekat "
                        f"dengan biaya yang diasumsikan ({self.assumed_cost_bps:.0f} "
                        f"bps). Turnover {self.costs.turnover_per_year:.0f}x setahun "
                        f"membuat hasil sangat sensitif terhadap biaya sebenarnya."
                    )

        if self.parameters is not None and self.parameters.n_runs > 1:
            if self.parameters.sharpe_win_rate < MIN_WIN_RATE:
                found.append(
                    f"Hanya {self.parameters.sharpe_win_rate:.0%} kombinasi parameter "
                    f"yang mengungguli pasif. Hasil yang baik bergantung pada "
                    f"pilihan parameter tertentu, bukan pada model itu sendiri."
                )

        if self.periods is not None and self.periods.n_runs > 1:
            if self.periods.win_rate < MIN_WIN_RATE:
                found.append(
                    f"Keunggulan nominal cuma muncul di {self.periods.win_rate:.0%} "
                    f"periode yang diuji. Hasil keseluruhan kemungkinan besar "
                    f"berasal dari satu rentang waktu tertentu."
                )

        if not np.isnan(self.deflated_sharpe):
            if self.deflated_sharpe < MIN_DEFLATED_SHARPE:
                found.append(
                    f"Setelah dikoreksi terhadap jumlah konfigurasi yang dicoba, "
                    f"peluang Sharpe ini benar-benar positif tinggal "
                    f"{self.deflated_sharpe:.0%}."
                )

        return found

    @property
    def n_trials_counted(self) -> int:
        """Percobaan yang ikut dihitung dalam koreksi multiple testing.

        Angka ini hampir selalu lebih kecil daripada jumlah percobaan yang
        sesungguhnya. Memilih model ini setelah mencoba model lain, memilih
        indikator ini setelah mencoba yang lain, memilih periode ini — tidak
        ada yang terhitung. Koreksinya hanya sebaik pembukuan yang menopangnya.
        """
        return self.parameters.n_runs if self.parameters is not None else 1

    def summary(self) -> dict[str, Any]:
        return {
            "baseline": dict(self.baseline),
            "deflated_sharpe": self.deflated_sharpe,
            "n_trials_counted": self.n_trials_counted,
            "rotation": self.rotation.summary() if self.rotation else None,
            "costs": self.costs.summary() if self.costs else None,
            "parameters": self.parameters.summary() if self.parameters else None,
            "periods": self.periods.summary() if self.periods else None,
            "concerns": self.concerns,
        }


def validate(
    experiment: Experiment,
    *,
    params: Mapping[str, Any] | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    cost_levels: Sequence[float] | None = None,
    n_rotations: int = 500,
    window_years: int = 3,
    step_years: int = 1,
    seed: int = 0,
) -> ValidationReport:
    """Jalankan seluruh pemeriksaan yang bisa dijalankan.

    Pemeriksaan yang tidak diminta akan dilewati: ``grid`` mengaktifkan
    stabilitas parameter, ``window_years`` yang muat di dalam rentang data
    mengaktifkan stabilitas periode. Uji rotasi dan sensitivitas biaya selalu
    dijalankan karena keduanya murah — tidak ada pelatihan ulang.

    Urutannya sengaja dari yang murah ke yang mahal, supaya masalah yang
    mencolok muncul sebelum sweep panjang dimulai.
    """
    # Seluruh pemeriksaan memakai jendela yang sama dengan panel utama: bagian
    # yang bisa diperdagangkan. Uji rotasi dan sensitivitas biaya menerima
    # deret yang sudah dipangkas, bukan yang mentah.
    prices, signals = tradable_slice(experiment.prices, experiment.signals(params))
    baseline = evaluate(prices, signals, experiment.backtest_kwargs)

    assumed = float(experiment.backtest_kwargs.get("cost_bps", 0.0)) + float(
        experiment.backtest_kwargs.get("slippage_bps", 0.0)
    )

    rotation = rotation_test(
        prices,
        signals,
        n_rotations=n_rotations,
        backtest_kwargs=experiment.backtest_kwargs,
        seed=seed,
    )
    costs = cost_sensitivity(
        prices,
        signals,
        levels=cost_levels if cost_levels is not None else _default_levels(assumed),
        backtest_kwargs=experiment.backtest_kwargs,
    )

    parameters = parameter_stability(experiment, grid) if grid else None
    periods = _period_stability_if_possible(
        experiment, window_years, step_years, params
    )

    from stocklab.backtest import run_backtest

    returns = run_backtest(prices, signals, **experiment.backtest_kwargs).returns
    trial_sharpes = parameters.sharpes() if parameters is not None else None
    deflated = deflated_sharpe_ratio(
        returns,
        n_trials=parameters.n_runs if parameters is not None else 1,
        trial_sharpes=trial_sharpes if trial_sharpes is not None and len(trial_sharpes) > 1 else None,
    )

    return ValidationReport(
        baseline=baseline,
        rotation=rotation,
        costs=costs,
        parameters=parameters,
        periods=periods,
        deflated_sharpe=deflated,
        assumed_cost_bps=assumed,
    )


def _default_levels(assumed: float) -> tuple[float, ...]:
    """Tingkat biaya yang mengepung asumsi user, bukan daftar tetap.

    Titik impas yang berguna adalah yang dekat dengan biaya yang benar-benar
    dibayar; rentang tetap bisa saja melewatkannya sama sekali.
    """
    base = assumed if assumed > 0 else 15.0
    multipliers = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    return tuple(round(base * m, 4) for m in multipliers)


def _period_stability_if_possible(
    experiment: Experiment,
    window_years: int,
    step_years: int,
    params: Mapping[str, Any] | None,
) -> StabilityReport | None:
    index = experiment.prices.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    span_years = (index[-1] - index[0]).days / 365.25
    if span_years < window_years:
        return None
    return period_stability(
        experiment,
        window_years=window_years,
        step_years=step_years,
        params=params,
    )
