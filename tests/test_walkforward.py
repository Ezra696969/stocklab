"""Uji validasi walk-forward.

Kelas :class:`TestNoLeakage` yang paling penting. Pengujiannya struktural, bukan
statistik: alih-alih memeriksa apakah hasilnya "terlihat wajar", setiap jendela
latih dicatat dan dibandingkan langsung dengan periode ujinya. Model yang
pernah melihat satu bar saja dari masa depan akan ketahuan, dan ketahuannya
pasti — bukan bergantung pada seed atau ambang yang dipilih enak sendiri.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.backtest import run_backtest, walk_forward_signals
from stocklab.features import FeatureConfig, build_features
from stocklab.models import build_model, direction_target
from tests.conftest import make_ohlcv


class RecordingAdapter:
    """Adapter palsu yang mencatat bar mana saja yang dilihatnya."""

    def __init__(self, log: list[dict]):
        self.log = log
        self.entry: dict | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.entry = {"train": X.index, "test": None}
        self.log.append(self.entry)

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        assert self.entry is not None, "predict dipanggil sebelum fit"
        self.entry["test"] = X.index
        return pd.Series(1.0, index=X.index)


def make_dataset(n: int = 600):
    ohlcv = make_ohlcv(n=n)
    features = build_features(
        ohlcv,
        [FeatureConfig("rsi", {"window": 14}), FeatureConfig("roc", {"window": 10})],
    )
    target = direction_target(ohlcv["close"])
    return ohlcv, features, target


def recording_factory(log: list[dict]):
    def factory() -> RecordingAdapter:
        return RecordingAdapter(log)

    return factory


class TestNoLeakage:
    def test_training_never_reaches_into_the_test_window(self):
        _, X, y = make_dataset()
        log: list[dict] = []

        walk_forward_signals(
            recording_factory(log), X, y, train_size=200, test_size=40, embargo=1
        )

        assert log, "tidak ada fold yang dilatih"
        for fold in log:
            assert fold["train"].max() < fold["test"].min()

    @pytest.mark.parametrize("embargo", [1, 5, 20])
    def test_embargo_gap_is_respected(self, embargo: int):
        """Label melihat ``horizon`` bar ke depan, jadi ujung data latih dibuang.

        Tanpa jarak ini, baris latih terakhir memuat informasi dari bar yang
        secara kronologis sudah berada di dalam periode uji.
        """
        _, X, y = make_dataset()
        log: list[dict] = []

        walk_forward_signals(
            recording_factory(log), X, y, train_size=200, test_size=40, embargo=embargo
        )

        positions = {timestamp: i for i, timestamp in enumerate(X.index)}
        for fold in log:
            gap = positions[fold["test"].min()] - positions[fold["train"].max()]
            assert gap >= embargo

    def test_a_model_is_built_fresh_for_every_fold(self):
        """Model yang dipakai ulang membawa keadaan masa depan ke masa lalu."""
        _, X, y = make_dataset()
        built: list[RecordingAdapter] = []
        log: list[dict] = []

        def factory() -> RecordingAdapter:
            adapter = RecordingAdapter(log)
            built.append(adapter)
            return adapter

        result = walk_forward_signals(factory, X, y, train_size=200, test_size=40)

        assert len(built) == result.n_fitted
        assert len({id(adapter) for adapter in built}) == len(built)

    def test_no_signal_before_the_first_training_window(self):
        _, X, y = make_dataset()
        result = walk_forward_signals(
            recording_factory([]), X, y, train_size=200, test_size=40
        )

        assert result.signals.iloc[:200].isna().all()
        assert result.signals.iloc[200:].notna().any()

    def test_test_windows_tile_the_remainder_without_overlap(self):
        _, X, y = make_dataset()
        log: list[dict] = []

        walk_forward_signals(
            recording_factory(log), X, y, train_size=200, test_size=40
        )

        covered = np.concatenate([fold["test"].to_numpy() for fold in log])
        assert len(covered) == len(set(covered)), "ada bar yang diuji dua kali"


class TestWindowing:
    def test_expanding_window_grows(self):
        _, X, y = make_dataset()
        log: list[dict] = []

        walk_forward_signals(
            recording_factory(log), X, y, train_size=200, test_size=40, expanding=True
        )

        sizes = [len(fold["train"]) for fold in log]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_rolling_window_stays_bounded(self):
        """Jendela bergulir berguna kalau perilaku pasar diduga berubah."""
        _, X, y = make_dataset()
        log: list[dict] = []

        walk_forward_signals(
            recording_factory(log), X, y, train_size=200, test_size=40, expanding=False
        )

        assert all(len(fold["train"]) <= 200 for fold in log)

    def test_test_size_sets_the_refit_cadence(self):
        _, X, y = make_dataset()

        frequent = walk_forward_signals(
            recording_factory([]), X, y, train_size=200, test_size=20
        )
        occasional = walk_forward_signals(
            recording_factory([]), X, y, train_size=200, test_size=100
        )

        assert frequent.n_folds > occasional.n_folds

    def test_result_reports_where_the_backtest_really_starts(self):
        _, X, y = make_dataset()
        result = walk_forward_signals(
            recording_factory([]), X, y, train_size=200, test_size=40
        )

        assert result.first_prediction == X.index[200]
        assert 0.0 < result.coverage < 1.0


class TestDataHandling:
    def test_warmup_rows_are_excluded_from_training(self):
        """Fitur NaN di awal tidak boleh masuk ke model."""
        _, X, y = make_dataset()
        log: list[dict] = []

        walk_forward_signals(
            recording_factory(log), X, y, train_size=200, test_size=40
        )

        for fold in log:
            assert X.loc[fold["train"]].notna().all().all()

    def test_bars_with_missing_features_get_no_signal(self):
        _, X, y = make_dataset()
        result = walk_forward_signals(
            recording_factory([]), X, y, train_size=200, test_size=40
        )

        warmup = X.isna().any(axis=1)
        assert result.signals[warmup].isna().all()

    def test_fold_with_a_single_label_class_is_skipped(self):
        """Pasar yang naik terus sepanjang jendela latih bisa terjadi."""
        _, X, _ = make_dataset()
        constant = pd.Series(1.0, index=X.index)

        result = walk_forward_signals(
            recording_factory([]), X, constant, train_size=200, test_size=40
        )

        assert result.n_fitted == 0
        assert result.signals.isna().all()
        assert result.n_folds > 0, "fold tetap dicatat meski dilewati"

    def test_mismatched_index_raises(self):
        _, X, y = make_dataset()
        with pytest.raises(ValueError, match="index yang sama"):
            walk_forward_signals(recording_factory([]), X, y.reset_index(drop=True))

    def test_negative_embargo_raises(self):
        _, X, y = make_dataset()
        with pytest.raises(ValueError, match="embargo"):
            walk_forward_signals(recording_factory([]), X, y, embargo=-1)


class TestEndToEnd:
    def test_real_model_produces_a_backtestable_signal(self):
        ohlcv, X, y = make_dataset(n=700)

        result = walk_forward_signals(
            lambda: build_model("logistic_regression"),
            X,
            y,
            train_size=250,
            test_size=60,
        )

        assert result.n_fitted > 0
        assert result.signals.dropna().between(0.0, 1.0).all()

        backtest = run_backtest(
            ohlcv["close"], result.signals, cost_bps=10.0, slippage_bps=5.0
        )
        assert backtest.equity.index.equals(ohlcv.index)
        assert backtest.equity.iloc[0] == pytest.approx(100_000.0)

    def test_out_of_sample_signal_is_not_a_perfect_oracle(self):
        """Walk-forward yang benar tidak akan menghasilkan model yang selalu benar.

        Kalau hit rate mendekati sempurna, hampir pasti ada informasi masa depan
        yang bocor ke pelatihan.
        """
        ohlcv, X, y = make_dataset(n=700)

        result = walk_forward_signals(
            lambda: build_model("random_forest", {"n_estimators": 50}),
            X,
            y,
            train_size=250,
            test_size=60,
        )

        predicted = result.signals.dropna()
        actual = y.loc[predicted.index]
        agreement = (predicted == actual).mean()

        assert agreement < 0.7, f"hit rate {agreement:.2f} terlalu tinggi untuk luar-sampel"
