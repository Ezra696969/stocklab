"""Uji metrik performa, termasuk koreksi multiple testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.backtest import run_backtest
from stocklab.backtest.metrics import (
    TRADING_DAYS,
    alpha,
    beta,
    cagr,
    deflated_sharpe_ratio,
    exposure,
    hit_rate,
    information_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    summarize,
    total_return,
    turnover_per_year,
)
from tests.test_engine import INITIAL, make_prices


def series(values: list[float] | np.ndarray) -> pd.Series:
    index = pd.date_range("2023-01-02", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=index)


class TestBasicMetrics:
    def test_total_return(self):
        assert total_return(series([100.0, 110.0, 150.0])) == pytest.approx(0.5)

    def test_cagr_doubling_in_one_year(self):
        """253 bar berarti 252 periode return, tepat satu tahun."""
        equity = series(np.linspace(100.0, 200.0, TRADING_DAYS + 1))
        assert cagr(equity) == pytest.approx(1.0)

    def test_cagr_doubling_in_two_years(self):
        equity = series(np.linspace(100.0, 200.0, 2 * TRADING_DAYS + 1))
        assert cagr(equity) == pytest.approx(2.0**0.5 - 1.0)

    def test_max_drawdown_of_rising_equity_is_zero(self):
        assert max_drawdown(series([100.0, 110.0, 120.0])) == pytest.approx(0.0)

    def test_max_drawdown_known_case(self):
        # Puncak 200, lembah 150 => -25%.
        assert max_drawdown(series([100.0, 200.0, 150.0, 180.0])) == pytest.approx(-0.25)

    def test_max_drawdown_uses_deepest_not_last(self):
        equity = series([100.0, 200.0, 100.0, 300.0, 270.0])
        assert max_drawdown(equity) == pytest.approx(-0.5)

    def test_sharpe_of_constant_returns_is_not_finite(self):
        """Volatilitas nol berarti Sharpe tidak terdefinisi, bukan tak hingga."""
        assert np.isnan(sharpe_ratio(series([0.0, 0.01, 0.01, 0.01])))

    def test_sharpe_scales_with_annualization(self):
        rng = np.random.default_rng(2)
        returns = series(np.concatenate(([0.0], rng.normal(0.001, 0.01, 500))))

        daily = sharpe_ratio(returns, periods_per_year=1)
        annual = sharpe_ratio(returns, periods_per_year=TRADING_DAYS)
        assert annual == pytest.approx(daily * np.sqrt(TRADING_DAYS))

    def test_first_bar_is_excluded_from_periods(self):
        """Bar pertama adalah titik awal equity, bukan periode investasi."""
        with_leading_zero = series([0.0, 0.01, -0.005, 0.02])
        without = series([0.01, -0.005, 0.02])
        # Dua-duanya membuang elemen pertama, jadi yang kedua kehilangan satu data.
        assert sharpe_ratio(with_leading_zero) != pytest.approx(sharpe_ratio(without))


class TestExposureMetrics:
    def test_hit_rate_ignores_cash_bars(self):
        returns = series([0.0, 0.01, -0.02, 0.0, 0.0, 0.03])
        weights = series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        # Bar kas dan bar ber-return nol dibuang: 2 dari 3 menguntungkan.
        assert hit_rate(returns, weights) == pytest.approx(2.0 / 3.0)

    def test_hit_rate_without_weights_uses_nonzero_returns(self):
        returns = series([0.0, 0.01, -0.02, 0.03, 0.04])
        assert hit_rate(returns) == pytest.approx(0.75)

    def test_exposure(self):
        weights = series([0.0, 1.0, 1.0, 0.0, 1.0])
        assert exposure(weights) == pytest.approx(0.75)

    def test_turnover_per_year(self):
        turnover = series(np.concatenate(([0.0], np.full(TRADING_DAYS, 2.0))))
        assert turnover_per_year(turnover) == pytest.approx(2.0 * TRADING_DAYS)


class TestRelativeMetrics:
    def test_beta_of_buy_and_hold_is_one(self):
        prices = make_prices()
        result = run_backtest(prices, np.ones(len(prices)), cost_bps=0.0)
        assert beta(result.returns, result.benchmark_returns) == pytest.approx(1.0)

    def test_alpha_of_buy_and_hold_is_zero(self):
        prices = make_prices()
        result = run_backtest(prices, np.ones(len(prices)), cost_bps=0.0)
        assert alpha(result.returns, result.benchmark_returns) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_beta_of_half_exposure_is_half(self):
        prices = make_prices()
        result = run_backtest(prices, np.full(len(prices), 0.5), cost_bps=0.0)
        assert beta(result.returns, result.benchmark_returns) == pytest.approx(0.5)

    def test_information_ratio_undefined_when_tracking_benchmark(self):
        prices = make_prices()
        result = run_backtest(prices, np.ones(len(prices)), cost_bps=0.0)
        # Tracking error nol: tidak ada return aktif untuk diukur.
        assert np.isnan(information_ratio(result.returns, result.benchmark_returns))

    def test_cash_strategy_has_zero_beta(self):
        prices = make_prices()
        result = run_backtest(prices, np.zeros(len(prices)), cost_bps=0.0)
        assert beta(result.returns, result.benchmark_returns) == pytest.approx(0.0)


class TestMultipleTesting:
    """Deflated Sharpe: semakin banyak dicoba, semakin tinggi ambangnya."""

    @staticmethod
    def good_returns(seed: int = 1, n: int = 750) -> pd.Series:
        rng = np.random.default_rng(seed)
        return series(np.concatenate(([0.0], rng.normal(0.0008, 0.01, n))))

    def test_probability_is_bounded(self):
        returns = self.good_returns()
        for n_trials in (1, 10, 100, 1000):
            value = deflated_sharpe_ratio(returns, n_trials=n_trials)
            assert 0.0 <= value <= 1.0

    def test_more_trials_lowers_confidence(self):
        """Inti fiturnya: menggeser slider 200 kali harus ada harganya."""
        returns = self.good_returns()

        few = deflated_sharpe_ratio(returns, n_trials=1)
        many = deflated_sharpe_ratio(returns, n_trials=200)
        very_many = deflated_sharpe_ratio(returns, n_trials=5000)

        assert few > many > very_many

    def test_single_trial_matches_probabilistic_sharpe(self):
        """Tanpa pencarian, tidak ada yang perlu dikoreksi."""
        returns = self.good_returns()
        assert deflated_sharpe_ratio(returns, n_trials=1) == pytest.approx(
            probabilistic_sharpe_ratio(returns)
        )

    def test_observed_trial_spread_is_used(self):
        """Sebaran Sharpe antar percobaan lebih informatif daripada jumlahnya."""
        returns = self.good_returns()

        narrow = deflated_sharpe_ratio(
            returns, n_trials=50, trial_sharpes=np.full(50, 0.5) + np.arange(50) * 1e-4
        )
        wide = deflated_sharpe_ratio(
            returns,
            n_trials=50,
            trial_sharpes=np.linspace(-2.0, 2.0, 50),
        )
        # Pencarian yang menghasilkan Sharpe tersebar luas gampang melahirkan
        # pemenang kebetulan, jadi ambangnya lebih tinggi.
        assert wide < narrow

    def test_zero_variance_returns_are_undefined(self):
        assert np.isnan(deflated_sharpe_ratio(series(np.zeros(100))))

    def test_short_series_is_undefined(self):
        assert np.isnan(deflated_sharpe_ratio(series([0.0, 0.01])))


class TestSummarize:
    def test_summary_contains_expected_keys(self):
        prices = make_prices()
        rng = np.random.default_rng(13)
        signals = rng.integers(0, 2, size=len(prices)).astype(float)
        result = run_backtest(
            prices, signals, initial_capital=INITIAL, cost_bps=10.0, slippage_bps=5.0
        )

        summary = result.summary(n_trials=25)

        expected = {
            "total_return",
            "cagr",
            "volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "hit_rate",
            "exposure",
            "turnover_per_year",
            "total_cost",
            "benchmark_total_return",
            "benchmark_cagr",
            "benchmark_sharpe",
            "benchmark_max_drawdown",
            "alpha",
            "beta",
            "information_ratio",
            "n_trials",
            "deflated_sharpe",
        }
        assert set(summary) == expected
        assert summary["n_trials"] == 25.0

    def test_summary_matches_standalone_functions(self):
        prices = make_prices()
        result = run_backtest(prices, np.ones(len(prices)), cost_bps=0.0)
        summary = summarize(result)

        assert summary["sharpe"] == pytest.approx(sharpe_ratio(result.returns))
        assert summary["max_drawdown"] == pytest.approx(max_drawdown(result.equity))
        assert summary["total_cost"] == pytest.approx(0.0)

    def test_buy_and_hold_summary_matches_its_benchmark(self):
        prices = make_prices()
        result = run_backtest(prices, np.ones(len(prices)), cost_bps=0.0)
        summary = summarize(result)

        assert summary["cagr"] == pytest.approx(summary["benchmark_cagr"])
        assert summary["sharpe"] == pytest.approx(summary["benchmark_sharpe"])
        assert summary["max_drawdown"] == pytest.approx(
            summary["benchmark_max_drawdown"]
        )
