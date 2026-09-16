"""Alur dari data sampai hasil backtest, tanpa menyentuh Streamlit.

Modul ini yang dipanggil tombol Jalankan, tapi tidak tahu apa-apa soal tombol.
Itu disengaja: alur inilah yang paling mudah salah, dan alur yang hanya bisa
dijalankan lewat antarmuka berarti hanya bisa diuji lewat antarmuka.

Urutannya tetap: muat harga, hitung indikator, bentuk label, hasilkan sinyal
luar-sampel lewat walk-forward, lalu backtest. Tidak ada tahap yang boleh
melompati tahap sebelumnya.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from stocklab.backtest import BacktestResult, WalkForwardResult, run_backtest
from stocklab.backtest.walkforward import walk_forward_signals
from stocklab.features import FeatureConfig, build_features, warmup_bars
from stocklab.models import ModelSpec, build_model, direction_target
from stocklab.models.base import PositionSizing
from stocklab.validation import Experiment

__all__ = ["RunConfig", "RunResult", "make_experiment", "run_pipeline"]


@dataclass(frozen=True)
class RunConfig:
    """Seluruh pilihan user untuk satu run."""

    features: tuple[FeatureConfig, ...]
    model_name: str = "random_forest"
    model_params: Mapping[str, Any] = field(default_factory=dict)

    # Sizing
    threshold: float = 0.5
    allow_short: bool = False
    sizing: PositionSizing = "binary"

    # Backtest
    initial_capital: float = 100_000.0
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    execution_lag: int = 1

    # Walk-forward
    horizon: int = 1
    train_size: int = 252
    test_size: int = 21
    expanding: bool = True

    @property
    def embargo(self) -> int:
        """Jarak latih-uji mengikuti horizon label, bukan angka terpisah.

        Label ``y[t]`` memuat informasi bar ``t + horizon``, jadi embargo yang
        lebih kecil dari horizon akan membocorkan sebagian periode uji ke data
        latih. Menjadikannya turunan, bukan pilihan, membuat kombinasi yang
        salah itu tidak bisa dibentuk sama sekali.
        """
        return self.horizon

    @property
    def backtest_kwargs(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "cost_bps": self.cost_bps,
            "slippage_bps": self.slippage_bps,
            "execution_lag": self.execution_lag,
        }


@dataclass(frozen=True)
class RunResult:
    """Semua yang dihasilkan satu run, siap dirender."""

    ohlcv: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    walkforward: WalkForwardResult
    backtest: BacktestResult
    metrics: dict[str, float]
    spec: ModelSpec
    warmup: int
    skipped: int = 0
    """Bar yang dibuang di depan karena model belum boleh berpendapat."""

    @property
    def equity_ratio(self) -> float:
        final = float(self.backtest.equity.iloc[-1])
        passive = float(self.backtest.benchmark_equity.iloc[-1])
        return final / passive if passive else float("nan")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features.columns)

    def period(self) -> dict[str, Any]:
        return {
            "start": str(self.ohlcv.index[0].date()),
            "end": str(self.ohlcv.index[-1].date()),
            "n_bars": len(self.ohlcv),
            "cost_bps": self.backtest.params.get("cost_bps", 0.0)
            + self.backtest.params.get("slippage_bps", 0.0),
        }

    def walkforward_summary(self) -> dict[str, Any]:
        first = self.walkforward.first_prediction
        return {
            "n_folds": self.walkforward.n_folds,
            "n_fitted": self.walkforward.n_fitted,
            "first_prediction": str(first.date()) if first is not None else None,
            "coverage": self.walkforward.coverage,
            "skipped_bars": self.skipped,
        }


def _adapter_factory(config: RunConfig) -> Callable[[], Any]:
    def factory():
        return build_model(
            config.model_name,
            config.model_params,
            threshold=config.threshold,
            allow_short=config.allow_short,
            sizing=config.sizing,
        )

    return factory


def _signal_builder(config: RunConfig, factory: Callable[[], Any] | None = None):
    """Fungsi penghasil sinyal yang dipakai run biasa maupun pemeriksaan validasi.

    Keduanya harus memakai jalur yang persis sama; validasi yang menjalankan
    alur berbeda dari yang divalidasi tidak memvalidasi apa pun.
    """

    def build(X: pd.DataFrame, y: pd.Series, params: Mapping[str, Any]) -> pd.Series:
        overrides = {**dict(config.model_params), **dict(params)}
        make = factory or _adapter_factory(replace(config, model_params=overrides))
        result = walk_forward_signals(
            make,
            X,
            y,
            train_size=config.train_size,
            test_size=config.test_size,
            embargo=config.embargo,
            expanding=config.expanding,
        )
        return result.signals

    return build


def run_pipeline(
    ohlcv: pd.DataFrame,
    config: RunConfig,
    adapter_factory: Callable[[], Any] | None = None,
) -> RunResult:
    """Jalankan satu eksperimen dari harga sampai equity curve.

    Parameters
    ----------
    ohlcv:
        Sudah dinormalisasi oleh :mod:`stocklab.data`.
    config:
        Pilihan user.
    adapter_factory:
        Pengganti model zoo, dipakai untuk model yang di-upload. Harus
        menghasilkan adapter baru tiap dipanggil, karena walk-forward melatih
        ulang di tiap fold.
    """
    if not config.features:
        raise ValueError("pilih minimal satu indikator sebelum menjalankan backtest")

    features = build_features(ohlcv, list(config.features))
    target = direction_target(ohlcv["close"], horizon=config.horizon)
    factory = adapter_factory or _adapter_factory(config)

    spec = factory().describe()
    if spec.is_pretrained:
        # Model terlatih tidak boleh dilatih ulang: bobotnya justru itu yang
        # diunggah user. Walk-forward tidak berlaku, jadi prediksi dijalankan
        # sekali untuk seluruh rentang, dan keabsahannya bergantung sepenuhnya
        # pada train_end yang dideklarasikan user.
        walkforward = _pretrained_signals(factory(), features, spec)
    else:
        walkforward = walk_forward_signals(
            factory,
            features,
            target,
            train_size=config.train_size,
            test_size=config.test_size,
            embargo=config.embargo,
            expanding=config.expanding,
        )
        if walkforward.n_fitted == 0:
            raise ValueError(
                "tidak ada fold yang bisa dilatih. Perpanjang rentang tanggal, "
                "atau kecilkan jendela latih."
            )

    # Backtest hanya pada jendela yang bisa diperdagangkan. Selama warmup dan
    # jendela latih pertama, model belum boleh berpendapat sementara buy & hold
    # sudah terisi penuh; memasukkan periode itu bukan membandingkan dua
    # keputusan, melainkan membandingkan satu keputusan dengan ketiadaan
    # keputusan. Keduanya kini berangkat dari modal yang sama di tanggal sama.
    tradable = _tradable_window(ohlcv, walkforward)
    backtest = run_backtest(
        tradable["close"],
        walkforward.signals.loc[tradable.index],
        **config.backtest_kwargs,
    )

    return RunResult(
        ohlcv=tradable,
        features=features.loc[tradable.index],
        target=target.loc[tradable.index],
        walkforward=walkforward,
        backtest=backtest,
        metrics=backtest.summary(),
        spec=spec,
        warmup=warmup_bars(features),
        skipped=len(ohlcv) - len(tradable),
    )


def _pretrained_signals(
    adapter: Any, features: pd.DataFrame, spec: ModelSpec
) -> WalkForwardResult:
    """Jalankan model terlatih sekali untuk seluruh rentang.

    Dibungkus sebagai :class:`WalkForwardResult` supaya sisa alur tidak perlu
    membedakan asal sinyalnya. Jumlah fold nol dan bukan satu, karena memang
    tidak ada pelatihan yang terjadi -- melaporkannya sebagai fold akan
    menyiratkan validasi yang tidak pernah dilakukan.

    Bar sebelum ``train_end`` dibuang kalau user menyebutkannya: di periode itu
    model sedang melihat data yang pernah dipelajarinya.
    """
    signals = adapter.predict_signal(features)

    if spec.train_end is not None:
        after_training = features.index > spec.train_end
        signals = signals.where(after_training)

    return WalkForwardResult(signals=signals, folds=())


def _tradable_window(
    ohlcv: pd.DataFrame, walkforward: WalkForwardResult
) -> pd.DataFrame:
    """Potong data ke bar pertama yang sinyalnya bisa dieksekusi.

    Dimulai satu bar sebelum prediksi pertama, karena sinyal di bar itulah yang
    menentukan posisi di bar berikutnya.
    """
    first = walkforward.first_prediction
    if first is None:
        return ohlcv
    position = max(0, ohlcv.index.get_loc(first) - 1)
    return ohlcv.iloc[position:]


def make_experiment(
    ohlcv: pd.DataFrame,
    config: RunConfig,
    adapter_factory: Callable[[], Any] | None = None,
) -> Experiment:
    """Bungkus alur yang sama agar bisa dijalankan ulang oleh modul validasi."""
    return Experiment(
        prices=ohlcv["close"],
        features=build_features(ohlcv, list(config.features)),
        target=direction_target(ohlcv["close"], horizon=config.horizon),
        build_signals=_signal_builder(config, adapter_factory),
        backtest_kwargs=config.backtest_kwargs,
    )
