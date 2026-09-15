"""Uji kontrak model, konversi sinyal, adapter, dan model zoo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.models import (
    MODELS,
    ModelAdapter,
    ModelSpec,
    SklearnAdapter,
    build_model,
    direction_target,
    forward_return,
    get_model_entry,
    infer_signal_kind,
    to_weights,
)
from tests.conftest import make_ohlcv


def series(values: list[float]) -> pd.Series:
    index = pd.date_range("2023-01-02", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=index)


class TestToWeights:
    """Perbatasan antara keluaran model dan bahasa yang dipahami engine."""

    def test_proba_binary_uses_threshold(self):
        result = to_weights(series([0.2, 0.5, 0.8]), "proba", threshold=0.5)
        np.testing.assert_allclose(result.to_numpy(), [0.0, 1.0, 1.0])

    def test_higher_threshold_makes_the_model_pickier(self):
        raw = series([0.55, 0.65, 0.75])
        lenient = to_weights(raw, "proba", threshold=0.5).sum()
        strict = to_weights(raw, "proba", threshold=0.7).sum()
        assert strict < lenient

    def test_proba_scaled_tracks_confidence(self):
        result = to_weights(
            series([0.5, 0.75, 1.0]), "proba", threshold=0.5, sizing="scaled"
        )
        np.testing.assert_allclose(result.to_numpy(), [0.0, 0.5, 1.0])

    def test_proba_long_only_never_goes_short(self):
        result = to_weights(series([0.0, 0.1, 0.4]), "proba", allow_short=False)
        assert result.ge(0.0).all()

    def test_proba_short_allowed(self):
        result = to_weights(
            series([0.1, 0.5, 0.9]), "proba", threshold=0.6, allow_short=True
        )
        np.testing.assert_allclose(result.to_numpy(), [-1.0, 0.0, 1.0])

    def test_class_labels(self):
        result = to_weights(series([1.0, 0.0, 1.0]), "class")
        np.testing.assert_allclose(result.to_numpy(), [1.0, 0.0, 1.0])

    def test_class_negative_labels_need_short_enabled(self):
        raw = series([1.0, -1.0])
        np.testing.assert_allclose(
            to_weights(raw, "class", allow_short=False).to_numpy(), [1.0, 0.0]
        )
        np.testing.assert_allclose(
            to_weights(raw, "class", allow_short=True).to_numpy(), [1.0, -1.0]
        )

    def test_expected_return_binary_uses_sign_only(self):
        result = to_weights(series([0.001, -0.002, 0.05]), "expected_return")
        np.testing.assert_allclose(result.to_numpy(), [1.0, 0.0, 1.0])

    def test_expected_return_scaled_is_bounded(self):
        """Expected return tidak punya satuan yang bisa langsung jadi bobot."""
        raw = series([0.001, 0.01, 0.5, -0.3])
        result = to_weights(raw, "expected_return", sizing="scaled", allow_short=True)

        assert result.between(-1.0, 1.0).all()
        assert result.iloc[2] > result.iloc[1] > result.iloc[0]

    def test_nan_stays_nan(self):
        """Bar yang modelnya belum berpendapat harus tetap kosong, bukan nol."""
        result = to_weights(series([0.8, np.nan, 0.2]), "proba")
        assert np.isnan(result.iloc[1])

    def test_index_is_preserved(self):
        raw = series([0.8, 0.2])
        assert to_weights(raw, "proba").index.equals(raw.index)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="signal_kind"):
            to_weights(series([0.5]), "ramalan")

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            to_weights(series([0.5]), "proba", threshold=1.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="panjangnya"):
            to_weights(np.array([0.5, 0.6]), "proba", index=pd.RangeIndex(3))


class TestModelSpec:
    def test_pretrained_without_train_end_is_flagged(self):
        """Aplikasi tidak punya cara tahu apa model sudah melihat periode uji."""
        spec = ModelSpec(name="m", label="M", requires_fit=False)
        assert spec.is_pretrained
        assert spec.leakage_is_unverifiable

    def test_declared_train_end_clears_the_flag(self):
        spec = ModelSpec(
            name="m", label="M", requires_fit=False, train_end=pd.Timestamp("2022-12-31")
        )
        assert not spec.leakage_is_unverifiable

    def test_model_trained_here_is_never_flagged(self):
        assert not ModelSpec(name="m", label="M", requires_fit=True).leakage_is_unverifiable


class TestTarget:
    def test_forward_return_looks_one_bar_ahead(self):
        result = forward_return(series([100.0, 110.0, 99.0]), horizon=1)
        np.testing.assert_allclose(result.to_numpy(), [0.1, -0.1, np.nan])

    def test_last_bars_have_no_future(self):
        result = forward_return(series([1.0] * 10), horizon=3)
        assert result.iloc[-3:].isna().all()

    def test_direction_is_binary(self):
        result = direction_target(series([100.0, 110.0, 99.0, 105.0]))
        np.testing.assert_allclose(result.to_numpy()[:3], [1.0, 0.0, 1.0])

    def test_threshold_ignores_moves_too_small_to_pay_for(self):
        prices = series([100.0, 100.05, 100.0])
        assert direction_target(prices, threshold=0.0).iloc[0] == 1.0
        assert direction_target(prices, threshold=0.01).iloc[0] == 0.0

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            forward_return(series([1.0, 2.0]), horizon=0)


class TestSklearnAdapter:
    @staticmethod
    def training_data(n: int = 300):
        rng = np.random.default_rng(4)
        index = pd.date_range("2023-01-02", periods=n, freq="B")
        X = pd.DataFrame(
            {"rsi_14": rng.uniform(10, 90, n), "roc_10": rng.normal(0, 2, n)},
            index=index,
        )
        # Label yang benar-benar bisa dipelajari, supaya test soal mekanika
        # tidak gagal gara-gara model kebingungan.
        y = pd.Series((X["rsi_14"] < 50).astype(float), index=index)
        return X, y

    def test_satisfies_the_protocol(self):
        adapter = build_model("logistic_regression")
        assert isinstance(adapter, ModelAdapter)

    def test_fit_then_predict_gives_weights(self):
        X, y = self.training_data()
        adapter = build_model("logistic_regression")
        adapter.fit(X, y)

        weights = adapter.predict_signal(X)
        assert weights.index.equals(X.index)
        assert weights.isin([0.0, 1.0]).all()

    def test_predict_before_fit_raises(self):
        X, _ = self.training_data()
        with pytest.raises(RuntimeError, match="belum dilatih"):
            build_model("random_forest").predict_signal(X)

    def test_feature_names_are_recovered_from_the_artifact(self):
        """Jalur Tier 2: artifact menyimpan sendiri fitur yang dibutuhkannya."""
        from sklearn.linear_model import LogisticRegression

        X, y = self.training_data()
        estimator = LogisticRegression(max_iter=500).fit(X, y)

        adapter = SklearnAdapter(estimator, name="uploaded", pretrained=True)
        assert adapter.describe().feature_names == ("rsi_14", "roc_10")

    def test_columns_are_reordered_to_match_training(self):
        """Urutan kolom yang tertukar diam-diam adalah bug paling sunyi."""
        from sklearn.linear_model import LogisticRegression

        X, y = self.training_data()
        estimator = LogisticRegression(max_iter=500).fit(X, y)
        adapter = SklearnAdapter(estimator, pretrained=True)

        expected = adapter.predict_signal(X)
        shuffled = adapter.predict_signal(X[["roc_10", "rsi_14"]])
        pd.testing.assert_series_equal(expected, shuffled)

    def test_missing_feature_raises(self):
        from sklearn.linear_model import LogisticRegression

        X, y = self.training_data()
        estimator = LogisticRegression(max_iter=500).fit(X, y)
        adapter = SklearnAdapter(estimator, pretrained=True)

        with pytest.raises(ValueError, match="butuh fitur"):
            adapter.predict_signal(X[["rsi_14"]])

    def test_pretrained_model_refuses_to_be_retrained(self):
        from sklearn.linear_model import LogisticRegression

        X, y = self.training_data()
        adapter = SklearnAdapter(
            LogisticRegression(max_iter=500).fit(X, y), pretrained=True
        )
        with pytest.raises(NotImplementedError, match="apa adanya"):
            adapter.fit(X, y)

    def test_positive_class_column_respects_class_order(self):
        """Model berlabel {-1, 1} menaruh kelas positif di tempat berbeda."""
        from sklearn.linear_model import LogisticRegression

        X, y = self.training_data()
        y_signed = y.replace({0.0: -1.0})
        estimator = LogisticRegression(max_iter=500).fit(X, y_signed)

        adapter = SklearnAdapter(estimator, pretrained=True, signal_kind="proba")
        weights = adapter.predict_signal(X)

        # Peluang kelas +1 harus sejalan dengan label, bukan kebalikannya.
        agreement = (weights == (y_signed > 0).astype(float)).mean()
        assert agreement > 0.9

    def test_params_are_introspected_when_not_supplied(self):
        """Jalur Tier 2: panel parameter digali dari get_params()."""
        from sklearn.ensemble import RandomForestClassifier

        adapter = SklearnAdapter(RandomForestClassifier(n_estimators=42))
        params = adapter.describe().params

        assert params["n_estimators"].default == 42
        assert "random_state" not in params, "parameter internal tidak perlu dilihat user"

    def test_signal_kind_is_inferred_from_capabilities(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import Ridge

        assert infer_signal_kind(RandomForestClassifier()) == "proba"
        assert infer_signal_kind(Ridge()) == "expected_return"

    def test_rejects_object_that_is_not_an_estimator(self):
        with pytest.raises(TypeError, match="fit dan predict"):
            SklearnAdapter(object())


class TestModelZoo:
    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_entry_builds_and_describes_itself(self, name: str):
        adapter = build_model(name)
        spec = adapter.describe()

        assert spec.name == name
        assert spec.label and spec.description
        assert spec.requires_fit
        assert spec.params, "model tanpa parameter tidak ada gunanya di panel"

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_numeric_parameters_have_curated_ranges(self, name: str):
        """Model bawaan kita sendiri yang pilih, jadi batasnya memang diketahui."""
        for param in build_model(name).describe().params.values():
            if param.kind in ("int", "float"):
                assert param.is_bounded, f"{name}.{param.name} tidak punya batas"
                assert param.low <= param.default <= param.high

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_entry_trains_and_predicts(self, name: str):
        X, y = TestSklearnAdapter.training_data()
        adapter = build_model(name)
        adapter.fit(X, y)

        weights = adapter.predict_signal(X)
        assert weights.notna().all()
        assert weights.between(0.0, 1.0).all()

    def test_displayed_defaults_are_the_ones_actually_used(self):
        entry = get_model_entry("random_forest")
        specs = entry.param_specs()
        assert specs["n_estimators"].default == entry.defaults["n_estimators"]

    def test_user_parameters_reach_the_estimator(self):
        adapter = build_model("random_forest", {"n_estimators": 25, "max_depth": 3})
        model = adapter.estimator.named_steps["model"]

        assert model.n_estimators == 25
        assert model.max_depth == 3

    def test_unspecified_parameters_fall_back_to_defaults(self):
        adapter = build_model("random_forest", {"n_estimators": 25})
        assert adapter.estimator.named_steps["model"].max_depth == 8

    def test_out_of_range_parameter_raises(self):
        with pytest.raises(ValueError, match="di atas batas"):
            build_model("random_forest", {"n_estimators": 99_999})

    def test_unknown_parameter_raises(self):
        with pytest.raises(ValueError, match="tidak dikenal"):
            build_model("random_forest", {"jumlah_pohon": 100})

    def test_unknown_model_lists_alternatives(self):
        with pytest.raises(KeyError, match="random_forest"):
            get_model_entry("randomforest")

    def test_features_are_scaled_before_the_model(self):
        """Tanpa scaler, SMA di angka ribuan menenggelamkan RSI 0-100."""
        adapter = build_model("logistic_regression")
        assert "scaler" in adapter.estimator.named_steps

    def test_ridge_produces_expected_return_signals(self):
        assert build_model("ridge").describe().signal_kind == "expected_return"


class TestIntegration:
    def test_zoo_model_runs_on_indicator_features(self):
        from stocklab.features import FeatureConfig, build_features

        ohlcv = make_ohlcv()
        features = build_features(
            ohlcv,
            [
                FeatureConfig("rsi", {"window": 14}),
                FeatureConfig("roc", {"window": 10}),
                FeatureConfig("rolling_volatility", {"window": 20}),
            ],
        )
        target = direction_target(ohlcv["close"])

        usable = features.notna().all(axis=1) & target.notna()
        adapter = build_model("gradient_boosting", {"n_estimators": 20})
        adapter.fit(features[usable], target[usable])

        weights = adapter.predict_signal(features[usable])
        assert weights.index.equals(features[usable].index)
        assert weights.between(0.0, 1.0).all()
