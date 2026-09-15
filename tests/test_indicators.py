"""Uji indikator teknikal.

Yang paling penting ada di :class:`TestCausality`. Sisanya memeriksa nilai
tertentu; kausalitas memeriksa properti yang membuat seluruh backtest bermakna.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.features import FEATURES, FeatureConfig, build_features
from stocklab.features import indicators as ind
from tests.conftest import make_ohlcv


def series(values: list[float]) -> pd.Series:
    index = pd.date_range("2023-01-02", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=index)


class TestCausality:
    """Tidak ada indikator yang boleh melihat ke depan.

    Metodenya langsung: hitung indikator pada seluruh data, lalu hitung ulang
    pada potongan awalnya saja. Kalau indikator itu kausal, bagian yang tumpang
    tindih harus sama persis. Kalau ada informasi masa depan yang merembes —
    lewat ``center=True``, backfill, atau normalisasi seluruh sampel — nilai
    lama akan bergeser saat data masa depan dibuang, dan test ini gagal.
    """

    @pytest.mark.parametrize("name", sorted(FEATURES))
    @pytest.mark.parametrize("cut", [150, 300])
    def test_registered_indicator_ignores_future_bars(self, name: str, cut: int):
        ohlcv = make_ohlcv()
        config = [FeatureConfig(name)]

        with_future = build_features(ohlcv, config).iloc[:cut]
        without_future = build_features(ohlcv.iloc[:cut], config)

        pd.testing.assert_frame_equal(with_future, without_future)

    def test_appending_a_bar_does_not_rewrite_history(self):
        """Bar baru boleh menambah nilai, tidak boleh mengubah yang sudah ada."""
        ohlcv = make_ohlcv(n=200)
        configs = [FeatureConfig(name) for name in sorted(FEATURES)]

        before = build_features(ohlcv.iloc[:-1], configs)
        after = build_features(ohlcv, configs).iloc[:-1]

        pd.testing.assert_frame_equal(before, after)

    def test_a_deliberately_leaky_indicator_is_caught(self):
        """Pagarnya sendiri harus terbukti bekerja.

        Rolling window terpusat adalah cara paling umum kebocoran masa depan
        masuk tanpa disadari. Test ini memastikan pemeriksaan di atas memang
        menangkapnya, bukan lolos karena kebetulan.
        """
        ohlcv = make_ohlcv(n=200)
        cut = 120

        def leaky(close: pd.Series) -> pd.Series:
            return close.rolling(11, center=True).mean()

        with_future = leaky(ohlcv["close"]).iloc[:cut]
        without_future = leaky(ohlcv["close"].iloc[:cut])

        with pytest.raises(AssertionError):
            pd.testing.assert_series_equal(with_future, without_future)


class TestMovingAverages:
    def test_sma_known_values(self):
        result = ind.sma(series([1, 2, 3, 4, 5]), window=3)
        expected = [np.nan, np.nan, 2.0, 3.0, 4.0]
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_sma_warmup_length(self):
        result = ind.sma(series([1, 2, 3, 4, 5]), window=3)
        assert result.isna().sum() == 2

    def test_ema_starts_at_first_observation(self):
        """``adjust=False`` berarti rekursi diawali nilai pertama."""
        result = ind.ema(series([10, 20, 30]), window=5)
        assert result.iloc[0] == pytest.approx(10.0)

    def test_ema_tracks_constant_series(self):
        result = ind.ema(series([7.0] * 30), window=10)
        np.testing.assert_allclose(result.to_numpy(), 7.0)

    def test_sma_names_include_window(self):
        assert ind.sma(series([1, 2, 3]), window=2).name == "sma_2"


class TestRSI:
    def test_saturates_at_100_when_only_rising(self):
        result = ind.rsi(series(list(range(1, 40))), window=14)
        assert result.dropna().eq(100.0).all()

    def test_saturates_at_zero_when_only_falling(self):
        result = ind.rsi(series(list(range(40, 1, -1))), window=14)
        assert result.dropna().eq(0.0).all()

    def test_stays_within_bounds(self):
        ohlcv = make_ohlcv()
        result = ind.rsi(ohlcv["close"], window=14).dropna()
        assert result.between(0.0, 100.0).all()

    def test_warmup_is_window_bars(self):
        """Satu bar hilang untuk diff, sisanya untuk memenuhi min_periods."""
        ohlcv = make_ohlcv(n=60)
        assert ind.rsi(ohlcv["close"], window=14).isna().sum() == 14

    def test_shorter_window_reacts_faster(self):
        ohlcv = make_ohlcv()
        fast = ind.rsi(ohlcv["close"], window=5).dropna()
        slow = ind.rsi(ohlcv["close"], window=30).dropna()
        assert fast.std() > slow.std()


class TestMACD:
    def test_histogram_is_macd_minus_signal(self):
        ohlcv = make_ohlcv()
        result = ind.macd(ohlcv["close"])
        np.testing.assert_allclose(
            result["hist"].to_numpy(),
            (result["macd"] - result["signal"]).to_numpy(),
        )

    def test_constant_price_gives_zero_macd(self):
        result = ind.macd(series([42.0] * 100))
        np.testing.assert_allclose(result["macd"].to_numpy(), 0.0, atol=1e-12)

    def test_fast_must_be_shorter_than_slow(self):
        with pytest.raises(ValueError, match="lebih kecil"):
            ind.macd(series([1.0] * 50), fast=26, slow=12)


class TestBands:
    def test_pct_b_is_half_at_the_middle_band(self):
        result = ind.bollinger(series([10.0] * 40), window=20)
        # Deviasi nol membuat lebar band nol, jadi pct_b tak terdefinisi.
        assert result["pct_b"].dropna().empty

    def test_pct_b_matches_position_within_the_band(self):
        """0 di band bawah, 1 di band atas, dan linier di antaranya."""
        ohlcv = make_ohlcv()
        result = ind.bollinger(ohlcv["close"], window=20)

        rebuilt = (ohlcv["close"] - result["lower"]) / (
            result["upper"] - result["lower"]
        )
        pd.testing.assert_series_equal(
            result["pct_b"].dropna(), rebuilt.dropna(), check_names=False
        )

    def test_pct_b_is_scale_free(self):
        """Menggandakan seluruh harga tidak boleh mengubah pct_b."""
        ohlcv = make_ohlcv()
        base = ind.bollinger(ohlcv["close"], window=20)["pct_b"]
        scaled = ind.bollinger(ohlcv["close"] * 1000.0, window=20)["pct_b"]

        np.testing.assert_allclose(
            base.dropna().to_numpy(), scaled.dropna().to_numpy(), rtol=1e-9
        )

    def test_wider_band_setting_gives_wider_band(self):
        ohlcv = make_ohlcv()
        narrow = ind.bollinger(ohlcv["close"], window=20, n_std=1.0)
        wide = ind.bollinger(ohlcv["close"], window=20, n_std=3.0)
        width_narrow = (narrow["upper"] - narrow["lower"]).dropna()
        width_wide = (wide["upper"] - wide["lower"]).dropna()
        assert (width_wide > width_narrow).all()

    def test_negative_std_multiplier_raises(self):
        with pytest.raises(ValueError, match="n_std"):
            ind.bollinger(series([1.0] * 30), n_std=-1.0)


class TestRangeIndicators:
    def test_atr_is_positive(self):
        ohlcv = make_ohlcv()
        assert ind.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna().gt(0).all()

    def test_atr_accounts_for_gaps(self):
        """True range harus memasukkan lompatan dari penutupan sebelumnya."""
        high = series([10.0, 25.0])
        low = series([9.0, 24.0])
        close = series([9.5, 24.5])
        # Rentang intraday cuma 1.0, tapi gap dari 9.5 ke 25.0 jauh lebih besar.
        result = ind.atr(high, low, close, window=1)
        assert result.iloc[1] == pytest.approx(15.5)

    def test_stochastic_is_100_at_a_new_high(self):
        high = series([10.0, 11.0, 12.0, 13.0])
        low = series([9.0, 10.0, 11.0, 12.0])
        close = series([9.5, 10.5, 11.5, 13.0])
        result = ind.stochastic_k(high, low, close, window=3)
        assert result.iloc[-1] == pytest.approx(100.0)

    def test_stochastic_stays_within_bounds(self):
        ohlcv = make_ohlcv()
        result = ind.stochastic_k(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
        assert result.between(0.0, 100.0).all()


class TestReturnsAndVolatility:
    def test_returns_known_values(self):
        result = ind.returns(series([100.0, 110.0, 99.0]), window=1)
        np.testing.assert_allclose(result.to_numpy(), [np.nan, 0.1, -0.1])

    def test_log_returns_are_additive(self):
        prices = series([100.0, 110.0, 121.0])
        single = ind.log_returns(prices, window=1).dropna()
        double = ind.log_returns(prices, window=2).dropna()
        assert single.sum() == pytest.approx(double.iloc[0])

    def test_roc_is_percent(self):
        result = ind.roc(series([100.0, 150.0]), window=1)
        assert result.iloc[1] == pytest.approx(50.0)

    def test_volatility_of_constant_price_is_zero(self):
        result = ind.rolling_volatility(series([5.0] * 60), window=20)
        assert result.dropna().eq(0.0).all()

    def test_volatility_annualization(self):
        ohlcv = make_ohlcv()
        daily = ind.rolling_volatility(ohlcv["close"], window=20, periods_per_year=1)
        annual = ind.rolling_volatility(ohlcv["close"], window=20, periods_per_year=252)
        np.testing.assert_allclose(
            annual.dropna().to_numpy(),
            daily.dropna().to_numpy() * np.sqrt(252),
        )


class TestValidation:
    @pytest.mark.parametrize(
        "fn", [ind.sma, ind.ema, ind.rsi, ind.returns, ind.roc, ind.rolling_volatility]
    )
    def test_zero_window_raises(self, fn):
        with pytest.raises(ValueError, match="window"):
            fn(series([1.0] * 30), window=0)

    def test_non_integer_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            ind.sma(series([1.0] * 30), window=3.5)
