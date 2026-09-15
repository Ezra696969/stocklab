"""Metrik performa untuk hasil backtest.

Sengaja melewati "total return" sebagai angka utama. Yang dilaporkan adalah
metrik risk-adjusted dan metrik relatif terhadap benchmark, karena equity curve
yang menang tanpa konteks drawdown dan turnover tidak mengatakan apa-apa.

Konvensi periode
----------------
Bar pertama pada hasil backtest selalu ber-return nol: itu titik awal equity,
bukan periode investasi. Semua fungsi di sini membuangnya lebih dulu lewat
:func:`_periodic`, sehingga ``T`` mencerminkan jumlah periode yang benar-benar
dijalani. Bar ber-return nol karena posisi kas **tidak** dibuang, karena itu
periode nyata.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "TRADING_DAYS",
    "annualized_volatility",
    "alpha",
    "beta",
    "cagr",
    "calmar_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "exposure",
    "hit_rate",
    "information_ratio",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize",
    "total_return",
    "turnover_per_year",
]

#: Jumlah hari perdagangan dalam setahun, dipakai untuk anualisasi.
TRADING_DAYS = 252


def _as_array(values: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _periodic(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Buang bar pertama, yang selalu nol karena hanya titik awal equity."""
    values = _as_array(returns)
    return values[1:] if len(values) > 1 else values


def _safe_std(values: np.ndarray) -> float:
    """Standar deviasi sampel; ``nan`` kalau datanya tidak cukup."""
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def total_return(equity: pd.Series) -> float:
    """Return total dari awal sampai akhir equity curve."""
    values = _as_array(equity)
    if len(values) == 0 or values[0] == 0:
        return float("nan")
    return float(values[-1] / values[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate.

    Lama periode dihitung dari jumlah bar dibagi ``periods_per_year``, bukan
    dari selisih tanggal kalender, agar konsisten dengan anualisasi metrik lain.
    """
    values = _as_array(equity)
    periods = len(values) - 1
    if periods <= 0 or values[0] <= 0 or values[-1] <= 0:
        return float("nan")
    years = periods / periods_per_year
    return float((values[-1] / values[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Volatilitas return yang dianualisasi."""
    std = _safe_std(_periodic(returns))
    return float(std * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
) -> float:
    """Sharpe ratio tahunan.

    ``risk_free_rate`` diberikan dalam satuan tahunan dan dibagi rata ke tiap
    periode.
    """
    excess = _periodic(returns) - risk_free_rate / periods_per_year
    std = _safe_std(excess)
    if not std or math.isnan(std):
        return float("nan")
    return float(np.mean(excess) / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
) -> float:
    """Sortino ratio tahunan; hanya deviasi sisi bawah yang dihukum."""
    excess = _periodic(returns) - risk_free_rate / periods_per_year
    if len(excess) < 2:
        return float("nan")
    downside = np.minimum(excess, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std == 0:
        return float("nan")
    return float(np.mean(excess) / downside_std * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Penurunan terdalam dari puncak, sebagai angka negatif."""
    values = _as_array(equity)
    if len(values) == 0:
        return float("nan")
    peak = np.maximum.accumulate(values)
    return float(np.min(values / peak - 1.0))


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR dibagi kedalaman drawdown terburuk."""
    drawdown = max_drawdown(equity)
    if drawdown == 0 or math.isnan(drawdown):
        return float("nan")
    return float(cagr(equity, periods_per_year) / abs(drawdown))


def hit_rate(returns: pd.Series, weights: pd.Series | None = None) -> float:
    """Proporsi periode menguntungkan di antara periode yang benar-benar eksposur.

    Bar saat posisi kas dikecualikan: bertahan di kas bukan tebakan yang benar
    atau salah, dan memasukkannya akan menggelembungkan angka ini untuk model
    yang jarang bertransaksi.
    """
    values = _periodic(returns)
    if weights is not None:
        held = _periodic(weights) != 0.0
        values = values[held]
    values = values[values != 0.0]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values > 0.0))


def exposure(weights: pd.Series) -> float:
    """Proporsi bar dengan posisi terbuka."""
    values = _periodic(weights)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values != 0.0))


def turnover_per_year(
    turnover: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Turnover tahunan rata-rata; satu round-trip penuh bernilai 2.0."""
    values = _periodic(turnover)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values) * periods_per_year)


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sensitivitas strategi terhadap benchmark."""
    strategy = _periodic(returns)
    bench = _periodic(benchmark_returns)
    if len(strategy) < 2:
        return float("nan")
    variance = float(np.var(bench, ddof=1))
    if variance == 0:
        return float("nan")
    covariance = float(np.cov(strategy, bench, ddof=1)[0, 1])
    return covariance / variance


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Return tahunan yang tidak dijelaskan oleh eksposur ke benchmark."""
    b = beta(returns, benchmark_returns)
    if math.isnan(b):
        return float("nan")
    strategy = _periodic(returns)
    bench = _periodic(benchmark_returns)
    return float((np.mean(strategy) - b * np.mean(bench)) * periods_per_year)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Return aktif dibagi tracking error, dianualisasi."""
    active = _periodic(returns) - _periodic(benchmark_returns)
    std = _safe_std(active)
    if not std or math.isnan(std):
        return float("nan")
    return float(np.mean(active) / std * math.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Koreksi multiple testing
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    returns: pd.Series, sharpe_benchmark: float = 0.0
) -> float:
    """Peluang Sharpe sesungguhnya melampaui ``sharpe_benchmark``.

    Keduanya dalam satuan per-periode, bukan tahunan. Memperhitungkan skewness
    dan kurtosis, karena return strategi jarang berdistribusi normal dan Sharpe
    dari deret berekor tebal lebih tidak bisa dipercaya daripada yang terlihat.

    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    """
    values = _periodic(returns)
    n = len(values)
    if n < 3:
        return float("nan")
    std = _safe_std(values)
    if not std or math.isnan(std):
        return float("nan")

    sharpe = float(np.mean(values) / std)
    skewness = float(stats.skew(values, bias=False))
    # Kurtosis non-excess: 3.0 untuk distribusi normal.
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))

    denominator = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        return float("nan")
    z = (sharpe - sharpe_benchmark) * math.sqrt(n - 1) / math.sqrt(denominator)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Sharpe tertinggi yang diperkirakan muncul dari ``n_trials`` percobaan nihil.

    Inilah ambang yang harus dilampaui strategi sungguhan. Kalau dicoba cukup
    banyak konfigurasi, salah satunya pasti terlihat bagus karena kebetulan.
    """
    if n_trials <= 1 or sharpe_std <= 0 or math.isnan(sharpe_std):
        return 0.0
    gamma = np.euler_gamma
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sharpe_std * ((1.0 - gamma) * a + gamma * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    trial_sharpes: Sequence[float] | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Sharpe setelah dikoreksi terhadap jumlah konfigurasi yang dicoba.

    Setiap kali user menggeser periode RSI lalu menjalankan ulang, itu satu
    percobaan tambahan. Geser cukup banyak kali, dan konfigurasi terbaik akan
    terlihat meyakinkan tanpa alasan apa pun. Nilai yang dikembalikan adalah
    peluang bahwa strategi ini benar-benar punya Sharpe positif, mengingat
    berapa kali pencarian dilakukan.

    Parameters
    ----------
    n_trials:
        Jumlah konfigurasi yang dievaluasi dalam pencarian ini.
    trial_sharpes:
        Sharpe **tahunan** dari percobaan-percobaan itu, kalau tersedia.
        Sebarannya jauh lebih informatif daripada sekadar jumlahnya. Kalau
        dikosongkan, dipakai aproksimasi asimtotik untuk varians estimator
        Sharpe, yang cenderung lebih longgar.
    """
    values = _periodic(returns)
    if len(values) < 3:
        return float("nan")
    std = _safe_std(values)
    if not std or math.isnan(std):
        return float("nan")
    sharpe = float(np.mean(values) / std)

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        # Skala per-periode, agar sebanding dengan sharpe di atas.
        observed = _as_array(trial_sharpes) / math.sqrt(periods_per_year)
        sharpe_std = _safe_std(observed)
    else:
        # Varians asimtotik estimator Sharpe di bawah hipotesis nihil.
        sharpe_std = math.sqrt((1.0 + sharpe**2 / 2.0) / (len(values) - 1))

    threshold = expected_max_sharpe(n_trials, sharpe_std)
    return probabilistic_sharpe_ratio(returns, sharpe_benchmark=threshold)


def summarize(
    result: Any,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
    n_trials: int = 1,
    trial_sharpes: Sequence[float] | None = None,
) -> dict[str, float]:
    """Kumpulkan metrik strategi, benchmark, dan relatifnya menjadi satu dict.

    ``result`` adalah :class:`~stocklab.backtest.engine.BacktestResult`. Dibuat
    generik agar modul metrik tidak perlu mengimpor engine.
    """
    equity = result.equity
    returns = result.returns
    bench_equity = result.benchmark_equity
    bench_returns = result.benchmark_returns

    return {
        # Strategi
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year, risk_free_rate),
        "sortino": sortino_ratio(returns, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, periods_per_year),
        "hit_rate": hit_rate(returns, result.weights),
        "exposure": exposure(result.weights),
        "turnover_per_year": turnover_per_year(result.turnover, periods_per_year),
        "total_cost": result.total_cost,
        # Benchmark pasif
        "benchmark_total_return": total_return(bench_equity),
        "benchmark_cagr": cagr(bench_equity, periods_per_year),
        "benchmark_sharpe": sharpe_ratio(
            bench_returns, periods_per_year, risk_free_rate
        ),
        "benchmark_max_drawdown": max_drawdown(bench_equity),
        # Relatif terhadap benchmark
        "alpha": alpha(returns, bench_returns, periods_per_year),
        "beta": beta(returns, bench_returns),
        "information_ratio": information_ratio(
            returns, bench_returns, periods_per_year
        ),
        # Koreksi multiple testing
        "n_trials": float(n_trials),
        "deflated_sharpe": deflated_sharpe_ratio(
            returns, n_trials, trial_sharpes, periods_per_year
        ),
    }
