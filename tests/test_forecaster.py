"""Uji peramal deret waktu (Tier 3).

Tidak ada test di sini yang mengunduh model atau menyentuh jaringan. Pipeline
disuntikkan sebagai objek palsu yang bentuk keluarannya sama dengan Chronos,
sehingga yang diuji adalah logika milik kita: penyusunan konteks, konversi
kuantil jadi peluang, dan kausalitas — bukan benar-tidaknya ramalan Amazon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Peramal butuh torch, yang berukuran ratusan megabyte. Test ini dilewati kalau
# extra "forecast" tidak terpasang, alih-alih menggagalkan seluruh koleksi.
torch = pytest.importorskip("torch")

from stocklab.models.base import ModelAdapter
from stocklab.models.forecaster import (
    FORECASTERS,
    PRICE_COLUMN,
    ForecastAdapter,
    _probability_above,
    build_forecaster,
    get_forecaster_entry,
    resolve_device,
)


class FakePipeline:
    """Peniru Chronos: mencatat apa yang dilihatnya, menjawab bentuk yang sama."""

    def __init__(self, drift: float = 0.0):
        self.drift = drift
        self.seen: list[torch.Tensor] = []
        self.calls = 0

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.calls += 1
        self.seen.extend(inputs)

        rows = []
        for context in inputs:
            anchor = float(context[-1])
            # Sebaran simetris di sekitar anchor, digeser oleh drift.
            spread = np.linspace(-0.02, 0.02, len(quantile_levels))
            rows.append([anchor + self.drift + s for s in spread])

        block = np.asarray(rows, dtype=np.float32)
        repeated = np.repeat(block[:, None, :], prediction_length, axis=1)
        return torch.tensor(repeated), torch.tensor(block[:, 0])


def price_frame(n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    series = np.log(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))))
    return pd.DataFrame({PRICE_COLUMN: series}, index=index)


def adapter(pipeline: FakePipeline, **kwargs) -> ForecastAdapter:
    return build_forecaster(
        "chronos_bolt_tiny",
        {"context_length": 64, "batch_size": 32, **kwargs.pop("params", {})},
        pipeline=pipeline,
        **kwargs,
    )


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(FORECASTERS))
    def test_entry_is_self_describing(self, name: str):
        entry = FORECASTERS[name]

        assert entry.name == name
        assert entry.model_id.startswith("amazon/")
        assert entry.label and entry.description
        assert entry.params_mb > 0, "ukuran unduhan harus disebut ke user"

    @pytest.mark.parametrize("name", sorted(FORECASTERS))
    def test_parameters_have_curated_ranges(self, name: str):
        for spec in FORECASTERS[name].param_specs().values():
            assert spec.is_bounded
            assert spec.low <= spec.default <= spec.high

    def test_unknown_forecaster_lists_alternatives(self):
        with pytest.raises(KeyError, match="chronos_bolt_tiny"):
            get_forecaster_entry("chronos-bolt-tiny")

    def test_unknown_parameter_is_refused(self):
        with pytest.raises(ValueError, match="tidak dikenal"):
            build_forecaster("chronos_bolt_tiny", {"temperature": 0.7})

    def test_out_of_range_parameter_is_refused(self):
        with pytest.raises(ValueError, match="di atas batas"):
            build_forecaster("chronos_bolt_tiny", {"context_length": 99_999})


class TestSpec:
    def test_declares_the_price_column_it_needs(self):
        spec = adapter(FakePipeline()).describe()

        assert spec.feature_names == (PRICE_COLUMN,)
        assert spec.signal_kind == "proba"

    def test_is_pretrained_and_never_refit(self):
        spec = adapter(FakePipeline()).describe()

        assert spec.is_pretrained
        assert spec.requires_fit is False

    def test_download_size_is_mentioned(self):
        assert "MB" in adapter(FakePipeline()).describe().description

    def test_refuses_to_be_trained(self):
        X = price_frame()
        with pytest.raises(NotImplementedError, match="zero-shot"):
            adapter(FakePipeline()).fit(X, pd.Series(0.0, index=X.index))


class TestProbabilityFromQuantiles:
    """Peluang naik dibaca dari posisi harga pada kurva kuantil ramalan."""

    LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

    def test_anchor_at_the_median_is_a_coin_flip(self):
        quantiles = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        assert _probability_above(1.0, quantiles, self.LEVELS) == pytest.approx(0.5)

    def test_anchor_below_the_forecast_means_likely_up(self):
        quantiles = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        assert _probability_above(0.5, quantiles, self.LEVELS) > 0.85

    def test_anchor_above_the_forecast_means_likely_down(self):
        quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _probability_above(2.0, quantiles, self.LEVELS) < 0.15

    def test_result_is_always_a_probability(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            quantiles = np.sort(rng.normal(0, 1, len(self.LEVELS)))
            value = _probability_above(rng.normal(0, 2), quantiles, self.LEVELS)
            assert 0.0 <= value <= 1.0

    def test_forecast_without_spread_gives_no_opinion(self):
        """Ramalan yang seluruh kuantilnya sama tidak punya dasar berpendapat."""
        flat = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        assert _probability_above(1.0, flat, self.LEVELS) == 0.5

    def test_unsorted_quantiles_are_repaired(self):
        """Kuantil yang tidak monoton akan mengacaukan interpolasi."""
        jagged = np.array([0.9, 1.1, 1.0, 1.3, 1.2])
        assert 0.0 <= _probability_above(1.05, jagged, self.LEVELS) <= 1.0


class TestPredict:
    def test_satisfies_the_protocol(self):
        assert isinstance(adapter(FakePipeline()), ModelAdapter)

    def test_warmup_bars_get_no_signal(self):
        """Bar yang riwayatnya belum sepanjang konteks tidak bisa diramal."""
        X = price_frame()
        weights = adapter(FakePipeline()).predict_signal(X)

        assert weights.iloc[:64].isna().all()
        assert weights.iloc[64:].notna().all()

    def test_context_never_includes_the_bar_being_predicted(self):
        """Kausalitas: ramalan untuk bar t hanya melihat bar sebelum t."""
        X = price_frame(n=100)
        pipeline = FakePipeline()
        adapter(pipeline).predict_signal(X)

        series = X[PRICE_COLUMN].to_numpy()
        for offset, context in enumerate(pipeline.seen):
            index = 64 + offset
            expected = series[index - 64 : index]
            np.testing.assert_allclose(context.numpy(), expected, rtol=1e-6)

    def test_upward_forecast_becomes_a_long_position(self):
        X = price_frame()
        weights = adapter(FakePipeline(drift=0.05)).predict_signal(X)

        assert weights.dropna().eq(1.0).all()

    def test_downward_forecast_stays_in_cash(self):
        X = price_frame()
        weights = adapter(FakePipeline(drift=-0.05)).predict_signal(X)

        assert weights.dropna().eq(0.0).all()

    def test_batching_does_not_change_the_answer(self):
        X = price_frame()
        one = adapter(FakePipeline(drift=0.001), params={"batch_size": 8})
        many = adapter(FakePipeline(drift=0.001), params={"batch_size": 256})

        pd.testing.assert_series_equal(
            one.predict_signal(X), many.predict_signal(X)
        )

    def test_missing_price_column_explains_the_fix(self):
        X = price_frame().rename(columns={PRICE_COLUMN: "rsi_14"})

        with pytest.raises(ValueError, match="panel Indikator"):
            adapter(FakePipeline()).predict_signal(X)

    def test_nan_in_context_skips_that_bar(self):
        X = price_frame()
        X.iloc[100, 0] = np.nan

        weights = adapter(FakePipeline(drift=0.05)).predict_signal(X)

        # Bar 100 sendiri tetap dapat sinyal: konteksnya [36:100], tidak memuat
        # dirinya. Yang kehilangan sinyal adalah 64 bar sesudahnya, karena
        # konteks merekalah yang memuat NaN itu.
        assert weights.iloc[100] == 1.0
        assert weights.iloc[101:165].isna().all()
        assert weights.iloc[165:].notna().all()

    def test_history_shorter_than_context_yields_nothing(self):
        X = price_frame(n=30)
        weights = adapter(FakePipeline()).predict_signal(X)

        assert weights.isna().all()


class TestMissingDependency:
    """Peramal ada di katalog walau pustakanya belum terpasang."""

    def test_error_names_the_command_that_fixes_it(self, monkeypatch):
        import builtins

        import stocklab.models.forecaster as module

        module._PIPELINE_CACHE.clear()
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "chronos"):
                raise ImportError(f"no module named {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)

        with pytest.raises(ImportError, match=r'pip install -e "\.\[forecast\]"'):
            module.load_pipeline("amazon/chronos-bolt-tiny")


class TestDeviceResolution:
    def test_explicit_choice_is_respected(self):
        assert resolve_device("cpu") == "cpu"

    def test_auto_never_returns_an_unusable_device(self):
        """Flag CUDA adalah klaim; yang dikembalikan harus device yang terbukti.

        GPU bisa terlihat oleh driver tapi punya compute capability yang tidak
        didukung build torch terpasang. Kegagalannya baru muncul jauh di dalam
        inferensi, saat backtest sudah berjalan lama.
        """
        chosen = resolve_device("auto")
        assert chosen in ("cpu", "cuda")

        if chosen == "cuda":
            probe = torch.zeros(4, 4, device="cuda")
            torch.mm(probe, probe).cpu()
