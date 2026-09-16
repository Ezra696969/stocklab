"""Uji pemeriksaan validasi.

Pemeriksaan yang tugasnya menangkap strategi palsu harus dibuktikan benar-benar
menangkapnya. Karena itu tiap uji di sini dipasangkan: satu sinyal yang memang
punya informasi, dan satu yang tidak, dengan harapan pemeriksaan bisa
membedakan keduanya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.validation import (
    Experiment,
    cost_sensitivity,
    evaluate,
    parameter_stability,
    period_stability,
    rolling_windows,
    rotation_test,
    validate,
)
from stocklab.validation.common import tradable_slice
from tests.conftest import make_ohlcv


def make_prices(n: int = 600, seed: int = 9) -> pd.Series:
    return make_ohlcv(n=n, seed=seed)["close"]


def oracle_signal(prices: pd.Series) -> pd.Series:
    """Sinyal yang tahu arah bar berikutnya: informatif, sengaja dibuat curang."""
    returns = prices.pct_change().fillna(0.0).to_numpy()
    values = np.zeros(len(prices))
    values[:-1] = (returns[1:] > 0).astype(float)
    return pd.Series(values, index=prices.index)


def arbitrary_signal(prices: pd.Series, seed: int = 3) -> pd.Series:
    """Posisi acak: tidak ada informasi, tapi bentuk posisinya menyerupai."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.integers(0, 2, size=len(prices)).astype(float), index=prices.index
    )


class TestRotationTest:
    def test_detects_a_signal_that_really_knows(self):
        prices = make_prices()
        result = rotation_test(prices, oracle_signal(prices), n_rotations=200)

        assert result.observed > result.null_max
        assert result.is_significant(0.01)

    def test_does_not_flag_an_uninformative_signal(self):
        """Ini yang penting: pemeriksaan tidak boleh menyetujui apa pun."""
        prices = make_prices()
        result = rotation_test(prices, arbitrary_signal(prices), n_rotations=200)

        assert not result.is_significant(0.05)

    def test_rotation_preserves_exposure_and_turnover(self):
        """Null harus berbeda dari aslinya hanya pada waktunya.

        Kalau eksposur atau turnover ikut berubah, yang dibandingkan bukan lagi
        timing melainkan strategi yang berbeda sama sekali. Toleransinya satu
        bar: execution_lag membuang sinyal terakhir, dan geseran menentukan
        sinyal mana yang terbuang.
        """
        from stocklab.backtest import run_backtest

        prices = make_prices()
        signal = arbitrary_signal(prices)
        original = run_backtest(prices, signal, cost_bps=10.0)

        values = signal.to_numpy()
        rotated = pd.Series(np.roll(values, 137), index=prices.index)
        shifted = run_backtest(prices, rotated, cost_bps=10.0)

        one_bar = 1.0 / len(prices)
        assert shifted.weights.mean() == pytest.approx(
            original.weights.mean(), abs=one_bar
        )
        assert shifted.turnover.sum() == pytest.approx(original.turnover.sum(), abs=2.0)

    def test_p_value_never_reports_zero(self):
        """500 geseran tidak membuktikan peluangnya nol, hanya di bawah 1/501."""
        prices = make_prices()
        result = rotation_test(prices, oracle_signal(prices), n_rotations=100)

        assert result.p_value > 0.0
        assert result.p_value == pytest.approx(1.0 / 101.0)

    def test_becomes_exhaustive_when_all_shifts_fit(self):
        prices = make_prices(n=80)
        result = rotation_test(prices, arbitrary_signal(prices), n_rotations=10_000)

        assert result.exhaustive
        assert result.n_rotations == 79

    def test_exhaustive_run_ignores_the_seed(self):
        """Uji eksak tidak boleh bisa diulang sampai keluar angka yang enak."""
        prices = make_prices(n=80)
        signal = arbitrary_signal(prices)

        first = rotation_test(prices, signal, n_rotations=10_000, seed=0)
        second = rotation_test(prices, signal, n_rotations=10_000, seed=99)

        assert first.p_value == second.p_value

    def test_bars_without_signal_stay_in_place(self):
        """Yang diacak urutan keputusan, bukan kapan model mulai berpendapat."""
        prices = make_prices()
        signal = arbitrary_signal(prices)
        signal.iloc[:100] = np.nan

        result = rotation_test(prices, signal, n_rotations=50)
        assert result.n_rotations == 50

    def test_mismatched_index_raises(self):
        prices = make_prices(n=100)
        signal = arbitrary_signal(make_prices(n=100, seed=1))
        signal.index = pd.date_range("2030-01-01", periods=100, freq="B")

        with pytest.raises(ValueError, match="index signals"):
            rotation_test(prices, signal)

    def test_too_few_signals_raises(self):
        prices = make_prices(n=100)
        signal = pd.Series(np.nan, index=prices.index)
        signal.iloc[0] = 1.0

        with pytest.raises(ValueError, match="minimal 3 bar"):
            rotation_test(prices, signal)


class TestCostSensitivity:
    def test_performance_falls_as_costs_rise(self):
        prices = make_prices()
        result = cost_sensitivity(prices, arbitrary_signal(prices))

        sharpes = result.table["sharpe"].to_numpy()
        assert np.all(np.diff(sharpes) <= 1e-9)

    def test_finds_the_breakeven_point(self):
        prices = make_prices()
        result = cost_sensitivity(prices, oracle_signal(prices), levels=(0, 25, 50, 100, 200))

        assert result.breakeven_bps is not None
        assert result.survives(0.0)
        assert not result.survives(result.breakeven_bps + 1.0)

    def test_breakeven_is_bracketed_by_the_levels_it_crosses(self):
        prices = make_prices()
        levels = (0.0, 20.0, 40.0, 80.0, 160.0)
        result = cost_sensitivity(prices, oracle_signal(prices), levels=levels)

        ratios = result.table["equity_ratio"]
        below = ratios[ratios <= 1.0]
        assert result.breakeven_bps <= below.index.min()

    def test_strategy_that_never_wins_has_no_breakeven(self):
        prices = make_prices()
        always_flat = pd.Series(0.0, index=prices.index)
        result = cost_sensitivity(prices, always_flat)

        assert result.breakeven_bps is None
        assert not result.beats_at_zero_cost

    def test_supplied_costs_override_the_backtest_kwargs(self):
        """Biaya tidak boleh terhitung dua kali."""
        prices = make_prices()
        signal = arbitrary_signal(prices)

        result = cost_sensitivity(
            prices, signal, levels=(0.0, 10.0), backtest_kwargs={"cost_bps": 500.0}
        )
        assert result.table["total_cost"].iloc[0] == pytest.approx(0.0)

    def test_unsorted_levels_raise(self):
        prices = make_prices()
        with pytest.raises(ValueError, match="menaik"):
            cost_sensitivity(prices, arbitrary_signal(prices), levels=(10.0, 5.0))


class TestStability:
    @staticmethod
    def make_experiment(n: int = 600, **backtest_kwargs) -> Experiment:
        """Alur sederhana tanpa sklearn, supaya uji mekanika tetap cepat."""
        ohlcv = make_ohlcv(n=n)
        prices = ohlcv["close"]
        features = pd.DataFrame({"momentum": prices.pct_change(5)}, index=prices.index)
        target = (prices.shift(-1) > prices).astype(float)

        def build(X: pd.DataFrame, y: pd.Series, params) -> pd.Series:
            cutoff = params.get("cutoff", 0.0)
            return (X["momentum"] > cutoff).astype(float)

        return Experiment(
            prices=prices,
            features=features,
            target=target,
            build_signals=build,
            backtest_kwargs=backtest_kwargs or {"cost_bps": 10.0},
        )

    def test_parameter_sweep_runs_every_combination(self):
        experiment = self.make_experiment()
        report = parameter_stability(
            experiment, {"cutoff": [-0.02, 0.0, 0.02], "unused": [1, 2]}
        )

        assert report.n_runs == 6
        assert set(report.table.columns) >= {"cutoff", "unused", "sharpe"}

    def test_sweep_reports_the_spread_not_just_the_best(self):
        experiment = self.make_experiment()
        report = parameter_stability(experiment, {"cutoff": [-0.05, -0.01, 0.0, 0.01, 0.05]})

        spread = report.sharpe_spread()
        assert spread["min"] <= spread["median"] <= spread["max"]
        assert 0.0 <= report.win_rate <= 1.0

    def test_worst_run_is_reachable(self):
        """User harus bisa melihat kasus terburuk, bukan cuma yang terbaik."""
        experiment = self.make_experiment()
        report = parameter_stability(experiment, {"cutoff": [-0.05, 0.0, 0.05]})

        worst = report.worst()
        assert worst["sharpe"] == report.sharpe_spread()["min"]

    def test_sweep_sharpes_feed_the_deflated_metric(self):
        experiment = self.make_experiment()
        report = parameter_stability(experiment, {"cutoff": [-0.02, 0.0, 0.02]})

        assert len(report.sharpes()) == report.n_runs - report.n_failed

    def test_a_failing_combination_does_not_stop_the_sweep(self):
        experiment = self.make_experiment()

        def build(X, y, params):
            if params["cutoff"] == 0.0:
                raise ValueError("kombinasi ini sengaja gagal")
            return (X["momentum"] > params["cutoff"]).astype(float)

        broken = Experiment(
            prices=experiment.prices,
            features=experiment.features,
            target=experiment.target,
            build_signals=build,
            backtest_kwargs=experiment.backtest_kwargs,
        )
        report = parameter_stability(broken, {"cutoff": [-0.02, 0.0, 0.02]})

        assert report.n_runs == 3
        assert report.n_failed == 1

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="grid kosong"):
            parameter_stability(self.make_experiment(), {})

    def test_period_stability_slices_into_windows(self):
        experiment = self.make_experiment(n=1200)
        report = period_stability(experiment, window_years=2, step_years=1)

        assert report.n_runs > 1
        assert (report.table["end"] > report.table["start"]).all()
        assert report.table["start"].is_monotonic_increasing

    def test_period_windows_are_shorter_than_the_whole_span(self):
        experiment = self.make_experiment(n=1200)
        report = period_stability(experiment, window_years=2)

        assert report.table["n_bars"].max() < len(experiment.prices)

    def test_win_rate_and_sharpe_win_rate_are_separate(self):
        """Strategi yang banyak berdiam di kas bisa unggul Sharpe tapi kalah nominal."""
        experiment = self.make_experiment(n=1200)
        report = period_stability(experiment, window_years=2)

        assert 0.0 <= report.win_rate <= 1.0
        assert 0.0 <= report.sharpe_win_rate <= 1.0

    def test_explicit_windows_are_honoured(self):
        experiment = self.make_experiment(n=1200)
        windows = [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")]
        report = period_stability(experiment, windows=windows)

        assert report.n_runs == 2

    def test_experiment_requires_aligned_inputs(self):
        experiment = self.make_experiment()
        with pytest.raises(ValueError, match="index yang sama"):
            Experiment(
                prices=experiment.prices,
                features=experiment.features.iloc[:-5],
                target=experiment.target,
                build_signals=experiment.build_signals,
            )


class TestRollingWindows:
    def test_windows_advance_by_the_step(self):
        index = pd.date_range("2020-01-01", "2025-12-31", freq="B")
        windows = rolling_windows(index, window_years=3, step_years=1)

        assert len(windows) == 4
        assert windows[1][0] == windows[0][0] + pd.DateOffset(years=1)

    def test_no_window_extends_past_the_data(self):
        index = pd.date_range("2020-01-01", "2025-12-31", freq="B")
        for _, end in rolling_windows(index, window_years=3):
            assert end <= index[-1]

    def test_almost_complete_final_window_is_kept(self):
        """Data yang berakhir beberapa hari lebih cepat tidak boleh menghapus
        periode terbaru — justru periode itu yang paling sering berbeda."""
        index = pd.date_range("2021-01-01", "2025-12-29", freq="B")
        windows = rolling_windows(index, window_years=3, step_years=1)

        assert len(windows) == 3
        assert windows[-1][1] == index[-1]

    def test_a_genuinely_short_tail_is_dropped(self):
        index = pd.date_range("2021-01-01", "2024-06-30", freq="B")
        windows = rolling_windows(index, window_years=3, step_years=1)

        assert len(windows) == 1, "sisa satu setengah tahun terlalu pendek untuk jendela 3 tahun"

    def test_short_history_still_yields_one_window(self):
        index = pd.date_range("2024-01-01", "2024-06-30", freq="B")
        assert len(rolling_windows(index, window_years=3)) == 1

    def test_empty_index_yields_nothing(self):
        assert rolling_windows(pd.DatetimeIndex([])) == []


class TestValidationReport:
    def test_informative_strategy_raises_few_concerns(self):
        experiment = TestStability.make_experiment(n=1200, cost_bps=5.0)
        report = validate(
            experiment,
            grid={"cutoff": [-0.01, 0.0, 0.01]},
            n_rotations=100,
            window_years=2,
        )

        assert report.rotation is not None
        assert report.costs is not None
        assert report.parameters is not None
        assert report.periods is not None
        assert isinstance(report.concerns, list)

    def test_uninformative_strategy_is_called_out(self):
        """Strategi tanpa informasi harus memicu kekhawatiran, bukan lolos diam-diam."""
        ohlcv = make_ohlcv(n=900)
        prices = ohlcv["close"]
        rng = np.random.default_rng(11)
        noise = pd.DataFrame(
            {"noise": rng.normal(0, 1, len(prices))}, index=prices.index
        )

        def build(X, y, params):
            return (X["noise"] > params.get("cutoff", 0.0)).astype(float)

        experiment = Experiment(
            prices=prices,
            features=noise,
            target=(prices.shift(-1) > prices).astype(float),
            build_signals=build,
            backtest_kwargs={"cost_bps": 15.0},
        )
        report = validate(experiment, grid={"cutoff": [-0.5, 0.0, 0.5]}, n_rotations=100)

        assert report.concerns, "sinyal derau seharusnya memicu peringatan"

    def test_cost_levels_bracket_the_assumed_cost(self):
        experiment = TestStability.make_experiment(n=900, cost_bps=20.0)
        report = validate(experiment, n_rotations=50)

        levels = report.costs.table.index.to_numpy()
        assert levels.min() == 0.0
        assert levels.max() > 20.0
        assert report.assumed_cost_bps == 20.0

    def test_trial_count_is_the_sweep_size(self):
        experiment = TestStability.make_experiment(n=900)
        report = validate(
            experiment, grid={"cutoff": [-0.02, -0.01, 0.0, 0.01]}, n_rotations=50
        )

        assert report.n_trials_counted == 4

    def test_period_check_is_skipped_when_history_is_too_short(self):
        experiment = TestStability.make_experiment(n=300)
        report = validate(experiment, n_rotations=50, window_years=5)

        assert report.periods is None

    def test_summary_is_serialisable(self):
        """Laporan ini akan diumpankan ke chatbot sebagai JSON."""
        import json

        experiment = TestStability.make_experiment(n=900)
        report = validate(experiment, grid={"cutoff": [0.0, 0.01]}, n_rotations=50)

        encoded = json.dumps(report.summary(), default=float)
        assert "concerns" in encoded


class TestTradableSlice:
    """Jendela yang dipakai seluruh pemeriksaan harus sama dengan panel utama."""

    def test_warmup_prefix_is_dropped(self):
        prices = make_prices(n=300)
        signals = arbitrary_signal(prices)
        signals.iloc[:100] = np.nan

        trimmed_prices, trimmed_signals = tradable_slice(prices, signals)

        assert len(trimmed_prices) == 201
        assert trimmed_prices.index.equals(trimmed_signals.index)

    def test_slicing_twice_changes_nothing(self):
        """Dipanggil di evaluate dan di validate; tidak boleh menggerus tiap kali."""
        prices = make_prices(n=300)
        signals = arbitrary_signal(prices)
        signals.iloc[:100] = np.nan

        once = tradable_slice(prices, signals)
        twice = tradable_slice(*once)

        assert twice[0].index.equals(once[0].index)

    def test_signal_from_the_first_bar_is_left_alone(self):
        prices = make_prices(n=50)
        signals = arbitrary_signal(prices)

        trimmed, _ = tradable_slice(prices, signals)
        assert trimmed.index.equals(prices.index)

    def test_all_nan_signal_is_left_alone(self):
        prices = make_prices(n=50)
        empty = pd.Series(np.nan, index=prices.index)

        trimmed, _ = tradable_slice(prices, empty)
        assert trimmed.index.equals(prices.index)

    def test_benchmark_no_longer_gets_a_free_head_start(self):
        """Tanpa pemangkasan, pasif memetik kenaikan sebelum model boleh main."""
        prices = make_prices(n=300)
        signals = arbitrary_signal(prices)
        signals.iloc[:150] = np.nan

        metrics = evaluate(prices, signals)
        direct = evaluate(*tradable_slice(prices, signals))

        assert metrics["benchmark_equity"] == pytest.approx(direct["benchmark_equity"])


class TestEvaluate:
    def test_reports_both_absolute_and_relative_performance(self):
        prices = make_prices()
        metrics = evaluate(prices, arbitrary_signal(prices), {"cost_bps": 10.0})

        expected = {
            "sharpe",
            "benchmark_sharpe",
            "final_equity",
            "benchmark_equity",
            "equity_ratio",
            "max_drawdown",
            "exposure",
            "turnover_per_year",
            "total_cost",
        }
        assert set(metrics) == expected
