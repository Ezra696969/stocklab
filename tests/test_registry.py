"""Uji registry indikator dan ParamSpec.

Registry inilah yang membuat panel tuning bisa merender dirinya sendiri, jadi
yang diuji bukan cuma perhitungannya, tapi juga apakah metadata tiap entri
cukup untuk membangun kontrol UI tanpa menebak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.features import (
    FEATURES,
    FeatureConfig,
    build_features,
    column_names,
    get_feature,
    warmup_bars,
)
from stocklab.params import ParamSpec, infer_param_specs, specs_from_callable
from tests.conftest import make_ohlcv


class TestRegistryContract:
    """Tiap entri harus lengkap; yang setengah jadi akan merusak UI."""

    @pytest.mark.parametrize("name", sorted(FEATURES))
    def test_entry_is_self_describing(self, name: str):
        spec = FEATURES[name]

        assert spec.name == name, "kunci registry harus sama dengan nama spec"
        assert spec.label and spec.description
        assert spec.inputs, "indikator harus menyebut kolom yang dibutuhkannya"
        assert all(column in ("open", "high", "low", "close", "volume") for column in spec.inputs)

    @pytest.mark.parametrize("name", sorted(FEATURES))
    def test_every_parameter_is_renderable(self, name: str):
        """Rentang indikator kita sendiri diketahui, jadi slider selalu boleh."""
        spec = FEATURES[name]

        assert spec.params, f"'{name}' tidak punya parameter yang bisa diatur"
        for param_name, param in spec.params.items():
            assert param.name == param_name
            assert param.is_editable
            assert param.is_bounded, "batas harus diisi, bukan dibiarkan ditebak UI"
            assert param.low <= param.default <= param.high
            assert param.display_label

    @pytest.mark.parametrize("name", sorted(FEATURES))
    def test_defaults_produce_usable_output(self, name: str):
        ohlcv = make_ohlcv()
        result = build_features(ohlcv, [FeatureConfig(name)])

        assert list(result.columns) == column_names(FeatureConfig(name))
        assert result.index.equals(ohlcv.index)
        assert result.notna().any().all(), "indikator tidak menghasilkan nilai sama sekali"

    def test_unknown_indicator_lists_alternatives(self):
        with pytest.raises(KeyError, match="rsi"):
            get_feature("rsl")


class TestBuildFeatures:
    def test_same_indicator_twice_with_different_windows(self):
        """SMA 20 melawan SMA 50 adalah kasus paling umum; tidak boleh bentrok."""
        ohlcv = make_ohlcv()
        result = build_features(
            ohlcv,
            [
                FeatureConfig("sma", {"window": 20}),
                FeatureConfig("sma", {"window": 50}),
            ],
        )

        assert list(result.columns) == ["sma_20", "sma_50"]
        assert not result["sma_20"].equals(result["sma_50"])

    def test_multi_output_indicator_expands_to_columns(self):
        ohlcv = make_ohlcv()
        result = build_features(ohlcv, [FeatureConfig("macd")])

        assert list(result.columns) == ["macd_12_26_9_macd", "macd_12_26_9_signal", "macd_12_26_9_hist"]

    def test_float_parameter_in_column_name(self):
        names = column_names(FeatureConfig("bollinger", {"n_std": 2.5}))
        assert names[0].startswith("bollinger_20_2p5")

    def test_empty_config_gives_empty_frame(self):
        ohlcv = make_ohlcv()
        result = build_features(ohlcv, [])

        assert result.empty or result.shape[1] == 0
        assert result.index.equals(ohlcv.index)

    def test_warmup_nans_are_kept(self):
        """Mengisi NaN akan memalsukan pendapat yang belum dimiliki indikator."""
        ohlcv = make_ohlcv()
        result = build_features(ohlcv, [FeatureConfig("sma", {"window": 50})])

        assert result["sma_50"].iloc[:49].isna().all()
        assert result["sma_50"].iloc[49:].notna().all()

    def test_missing_input_column_raises(self):
        close_only = make_ohlcv()[["close"]]
        with pytest.raises(ValueError, match="butuh kolom"):
            build_features(close_only, [FeatureConfig("atr")])

    def test_unknown_parameter_raises(self):
        ohlcv = make_ohlcv()
        with pytest.raises(ValueError, match="tidak dikenal"):
            build_features(ohlcv, [FeatureConfig("rsi", {"periode": 14})])

    def test_out_of_range_parameter_raises(self):
        ohlcv = make_ohlcv()
        with pytest.raises(ValueError, match="di atas batas"):
            build_features(ohlcv, [FeatureConfig("rsi", {"window": 5000})])


class TestWarmup:
    def test_warmup_is_the_slowest_indicator(self):
        ohlcv = make_ohlcv()
        features = build_features(
            ohlcv,
            [
                FeatureConfig("sma", {"window": 10}),
                FeatureConfig("sma", {"window": 60}),
            ],
        )
        assert warmup_bars(features) == 59

    def test_no_features_means_no_warmup(self):
        ohlcv = make_ohlcv()
        assert warmup_bars(build_features(ohlcv, [])) == 0

    def test_indicator_longer_than_history(self):
        ohlcv = make_ohlcv(n=30)
        features = build_features(ohlcv, [FeatureConfig("sma", {"window": 200})])
        assert warmup_bars(features) == 30


class TestParamSpec:
    def test_coerce_casts_to_declared_type(self):
        spec = ParamSpec("window", "int", 14, 2, 100, 1)
        assert spec.coerce("20") == 20
        assert isinstance(spec.coerce(20.0), int)

    def test_coerce_rejects_out_of_range(self):
        spec = ParamSpec("window", "int", 14, 2, 100, 1)
        with pytest.raises(ValueError, match="di bawah batas"):
            spec.coerce(1)

    def test_clamp_pulls_to_the_edge(self):
        spec = ParamSpec("window", "int", 14, 2, 100, 1)
        assert spec.clamp(5000) == 100
        assert spec.clamp(-3) == 2

    def test_choice_rejects_unlisted_value(self):
        spec = ParamSpec("mode", "choice", "wilder", choices=("wilder", "cutler"))
        assert spec.coerce("cutler") == "cutler"
        with pytest.raises(ValueError, match="bukan pilihan"):
            spec.coerce("sma")

    def test_display_label_falls_back_to_name(self):
        assert ParamSpec("n_estimators", "int", 100).display_label == "N estimators"

    def test_unbounded_spec_is_not_slider_material(self):
        spec = ParamSpec("alpha", "float", 0.1)
        assert spec.is_editable
        assert not spec.is_bounded


class TestIntrospection:
    """Jalur yang nanti mengisi panel parameter model secara otomatis."""

    def test_infers_kind_from_defaults(self):
        specs = infer_param_specs(
            {"n_estimators": 100, "learning_rate": 0.1, "verbose": False}
        )

        assert specs["n_estimators"].kind == "int"
        assert specs["learning_rate"].kind == "float"
        assert specs["verbose"].kind == "bool"

    def test_bool_is_not_mistaken_for_int(self):
        """``bool`` adalah subclass ``int``; urutan pengecekan penting."""
        assert infer_param_specs({"flag": True})["flag"].kind == "bool"

    def test_ranges_are_never_invented(self):
        """Tanpa informasi batas, UI harus dapat kotak angka, bukan slider palsu."""
        spec = infer_param_specs({"n_estimators": 100})["n_estimators"]
        assert spec.low is None and spec.high is None
        assert not spec.is_bounded

    def test_overrides_win(self):
        override = ParamSpec("n_estimators", "int", 100, 10, 500, 10)
        specs = infer_param_specs(
            {"n_estimators": 100}, overrides={"n_estimators": override}
        )
        assert specs["n_estimators"].is_bounded

    def test_skipped_parameters_are_hidden(self):
        specs = infer_param_specs(
            {"n_estimators": 100, "random_state": 0}, skip=["random_state"]
        )
        assert "random_state" not in specs

    def test_string_without_choices_is_opaque(self):
        """Menebak opsi yang sah sama saja mengarang API orang lain."""
        specs = infer_param_specs({"criterion": "gini"})
        assert specs["criterion"].kind == "opaque"
        assert not specs["criterion"].is_editable

    def test_string_with_choices_is_a_dropdown(self):
        specs = infer_param_specs(
            {"criterion": "gini"}, choices={"criterion": ["gini", "entropy"]}
        )
        assert specs["criterion"].kind == "choice"
        assert specs["criterion"].choices == ("gini", "entropy")

    def test_callable_introspection_skips_data_arguments(self):
        """Deret harga adalah masukan, bukan sesuatu yang diatur user."""
        from stocklab.features import indicators

        specs = specs_from_callable(indicators.bollinger)

        assert set(specs) == {"window", "n_std"}
        assert specs["window"].default == 20
        assert specs["n_std"].default == 2.0
