"""Uji engine backtest.

Empat invariant di PLAN.md sec. 5 ada di kelas :class:`TestInvariants`. Kalau
keempatnya lulus, mekanika engine bisa dipercaya tanpa perlu melatih model apa
pun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.backtest import run_backtest, shift_signals
from stocklab.backtest.engine import BPS

INITIAL = 100_000.0


def make_prices(n: int = 500, seed: int = 7) -> pd.Series:
    """Random walk geometrik yang deterministik."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0004, scale=0.012, size=n)
    prices = 100.0 * np.exp(np.cumsum(steps))
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.Series(prices, index=index, name="close")


def flat_prices(n: int = 100, value: float = 50.0) -> pd.Series:
    """Harga konstan: mengisolasi biaya dari pergerakan pasar."""
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.Series(np.full(n, value), index=index, name="close")


class TestInvariants:
    """Keempat invariant yang membuktikan mekanika engine benar."""

    def test_full_exposure_equals_buy_and_hold(self):
        """1. Sinyal all-ones tanpa biaya harus identik dengan buy & hold.

        Dibandingkan dengan bentuk tertutup yang dihitung langsung dari harga,
        bukan dengan benchmark milik engine, supaya bukan sekadar membandingkan
        engine dengan dirinya sendiri.
        """
        prices = make_prices()
        result = run_backtest(
            prices, np.ones(len(prices)), initial_capital=INITIAL, cost_bps=0.0
        )

        expected = INITIAL * prices.to_numpy() / prices.iloc[0]
        np.testing.assert_allclose(result.equity.to_numpy(), expected, rtol=1e-12)

    def test_flat_signal_stays_in_cash(self):
        """2. Sinyal all-zeros harus mengendap datar di modal awal."""
        prices = make_prices()
        result = run_backtest(
            prices, np.zeros(len(prices)), initial_capital=INITIAL, cost_bps=25.0
        )

        np.testing.assert_allclose(
            result.equity.to_numpy(), np.full(len(prices), INITIAL), rtol=1e-12
        )
        assert result.total_cost == pytest.approx(0.0)

    @pytest.mark.parametrize("lag", [0, 1, 2, 5])
    def test_lag_is_equivalent_to_shifting_signals(self, lag: int):
        """3. Menjalankan dengan lag=k sama dengan menggeser sinyal k bar lalu lag=0.

        Ini yang membuktikan lag benar-benar sebuah pergeseran, bukan sekadar
        parameter yang diterima lalu diabaikan.
        """
        prices = make_prices()
        rng = np.random.default_rng(11)
        signals = rng.integers(0, 2, size=len(prices)).astype(float)

        lagged = run_backtest(
            prices, signals, initial_capital=INITIAL, cost_bps=5.0, execution_lag=lag
        )
        pre_shifted = run_backtest(
            prices,
            shift_signals(signals, lag),
            initial_capital=INITIAL,
            cost_bps=5.0,
            execution_lag=0,
        )

        np.testing.assert_allclose(
            lagged.equity.to_numpy(), pre_shifted.equity.to_numpy(), rtol=1e-12
        )

    @pytest.mark.parametrize("n_round_trips", [1, 3, 10])
    def test_round_trip_costs(self, n_round_trips: int):
        """4. N round-trip harus menggerus equity sebesar 2N kali biaya.

        Harga dibuat konstan agar yang tersisa hanya efek biaya.
        """
        cost_bps = 20.0
        prices = flat_prices(n=4 * n_round_trips + 10)

        signals = np.zeros(len(prices))
        for i in range(n_round_trips):
            # Masuk, tahan satu bar, keluar: turnover 1.0 di tiap arah.
            signals[2 + 4 * i : 4 + 4 * i] = 1.0

        result = run_backtest(
            prices, signals, initial_capital=INITIAL, cost_bps=cost_bps
        )

        assert result.turnover.sum() == pytest.approx(2.0 * n_round_trips)
        cost_frac = cost_bps * BPS
        expected = INITIAL * (1.0 - cost_frac) ** (2 * n_round_trips)
        assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)


class TestLookAhead:
    """Lag harus benar-benar mencegah sinyal melihat masa depan."""

    def test_oracle_signal_captures_next_bar(self):
        """Sinyal yang tahu return besok akan menang telak lewat lag=1.

        Bukan cara memakai engine, tapi ini memastikan bar mana yang dipanen
        oleh sinyal di bar t memang bar t+1, sesuai konvensi yang ditulis.
        """
        prices = make_prices()
        returns = prices.pct_change().fillna(0.0).to_numpy()

        oracle = np.zeros(len(prices))
        oracle[:-1] = (returns[1:] > 0).astype(float)

        result = run_backtest(prices, oracle, initial_capital=INITIAL, cost_bps=0.0)

        # Hanya memanen bar naik, jadi equity tidak pernah turun.
        assert result.returns.min() >= 0.0
        assert result.equity.iloc[-1] > result.benchmark_equity.iloc[-1]

    def test_contemporaneous_signal_has_no_foresight(self):
        """Sinyal dari return hari ini, dieksekusi besok, tidak punya keunggulan itu.

        Sinyal yang sama persis, hanya bergeser satu bar, harus kehilangan
        seluruh keajaibannya. Kalau test ini gagal, ada kebocoran di engine.
        """
        prices = make_prices()
        returns = prices.pct_change().fillna(0.0).to_numpy()
        momentum = (returns > 0).astype(float)

        result = run_backtest(prices, momentum, initial_capital=INITIAL, cost_bps=0.0)

        assert result.returns.min() < 0.0


class TestBenchmark:
    """Garis pasif harus adil terhadap garis strategi."""

    def test_benchmark_is_buy_and_hold(self):
        prices = make_prices()
        result = run_backtest(
            prices, np.zeros(len(prices)), initial_capital=INITIAL, cost_bps=0.0
        )

        expected = INITIAL * prices.to_numpy() / prices.iloc[0]
        np.testing.assert_allclose(
            result.benchmark_equity.to_numpy(), expected, rtol=1e-12
        )

    def test_benchmark_pays_the_same_entry_cost(self):
        """Keduanya bayar biaya masuk, jadi selisihnya murni keputusan model."""
        prices = make_prices()
        always_in = run_backtest(
            prices, np.ones(len(prices)), initial_capital=INITIAL, cost_bps=30.0
        )

        np.testing.assert_allclose(
            always_in.equity.to_numpy(),
            always_in.benchmark_equity.to_numpy(),
            rtol=1e-12,
        )

    def test_both_curves_start_at_initial_capital(self):
        prices = make_prices()
        result = run_backtest(prices, np.ones(len(prices)), initial_capital=INITIAL)

        assert result.equity.iloc[0] == pytest.approx(INITIAL)
        assert result.benchmark_equity.iloc[0] == pytest.approx(INITIAL)


class TestAccounting:
    """Pembukuan biaya harus bisa direkonsiliasi dengan equity curve."""

    def test_costs_reconcile_with_equity(self):
        prices = flat_prices(n=60)
        rng = np.random.default_rng(3)
        signals = rng.integers(0, 2, size=len(prices)).astype(float)

        result = run_backtest(
            prices, signals, initial_capital=INITIAL, cost_bps=15.0, slippage_bps=5.0
        )

        drop = INITIAL - result.equity.iloc[-1]
        assert result.total_cost == pytest.approx(drop, rel=1e-10)

    def test_slippage_and_cost_are_additive(self):
        prices = make_prices(n=200)
        rng = np.random.default_rng(5)
        signals = rng.integers(0, 2, size=len(prices)).astype(float)

        combined = run_backtest(prices, signals, cost_bps=10.0, slippage_bps=5.0)
        single = run_backtest(prices, signals, cost_bps=15.0, slippage_bps=0.0)

        np.testing.assert_allclose(
            combined.equity.to_numpy(), single.equity.to_numpy(), rtol=1e-12
        )


class TestWeights:
    """Bobot pecahan dan short diperlakukan sebagaimana mestinya."""

    def test_half_weight_earns_half_the_move(self):
        prices = make_prices(n=200)
        half = run_backtest(prices, np.full(len(prices), 0.5), cost_bps=0.0)
        full = run_backtest(prices, np.ones(len(prices)), cost_bps=0.0)

        np.testing.assert_allclose(
            half.returns.to_numpy(), 0.5 * full.returns.to_numpy(), rtol=1e-12
        )

    def test_short_position_inverts_returns(self):
        prices = make_prices(n=200)
        short = run_backtest(
            prices, np.full(len(prices), -1.0), cost_bps=0.0, max_leverage=1.0
        )
        long = run_backtest(prices, np.ones(len(prices)), cost_bps=0.0)

        np.testing.assert_allclose(
            short.returns.to_numpy(), -long.returns.to_numpy(), rtol=1e-12
        )

    def test_nan_signal_is_treated_as_flat(self):
        """NaN saat warmup indikator berarti model belum berpendapat."""
        prices = make_prices(n=100)
        signals = np.ones(len(prices))
        signals[:20] = np.nan

        result = run_backtest(prices, signals, cost_bps=0.0)

        assert result.weights.iloc[:20].eq(0.0).all()
        assert result.weights.iloc[25:].eq(1.0).all()


class TestValidation:
    """Masukan yang salah harus gagal keras, bukan diam-diam dibetulkan."""

    def test_mismatched_index_raises(self):
        prices = make_prices(n=50)
        signals = pd.Series(
            np.ones(50), index=pd.date_range("2024-01-01", periods=50, freq="B")
        )

        with pytest.raises(ValueError, match="index signals"):
            run_backtest(prices, signals)

    def test_wrong_length_raises(self):
        prices = make_prices(n=50)
        with pytest.raises(ValueError, match="sepanjang 50 bar"):
            run_backtest(prices, np.ones(49))

    def test_nan_price_raises(self):
        prices = make_prices(n=50)
        prices.iloc[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            run_backtest(prices, np.ones(50))

    def test_weight_above_max_leverage_raises(self):
        prices = make_prices(n=50)
        with pytest.raises(ValueError, match="max_leverage"):
            run_backtest(prices, np.full(50, 2.5), max_leverage=1.0)

    def test_negative_lag_raises(self):
        prices = make_prices(n=50)
        with pytest.raises(ValueError, match="execution_lag"):
            run_backtest(prices, np.ones(50), execution_lag=-1)

    def test_negative_cost_raises(self):
        prices = make_prices(n=50)
        with pytest.raises(ValueError, match="tidak boleh negatif"):
            run_backtest(prices, np.ones(50), cost_bps=-1.0)

    def test_series_index_is_preserved(self):
        prices = make_prices(n=50)
        result = run_backtest(prices, np.ones(50))

        assert result.equity.index.equals(prices.index)
        assert result.weights.index.equals(prices.index)
        assert result.params["n_bars"] == 50
        assert result.params["execution_lag"] == 1


class TestShiftSignals:
    def test_shift_pads_with_zeros(self):
        signals = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            shift_signals(signals, 2), np.array([0.0, 0.0, 1.0, 2.0])
        )

    def test_shift_zero_is_identity(self):
        signals = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(shift_signals(signals, 0), signals)

    def test_shift_beyond_length_is_all_zeros(self):
        signals = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(shift_signals(signals, 5), np.zeros(3))
