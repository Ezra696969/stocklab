"""Validasi walk-forward: melatih di masa lalu, memprediksi ke depan.

Inilah bagian yang membedakan backtest yang berarti dari yang tidak. Melatih
model pada seluruh rentang lalu memprediksi rentang yang sama akan menghasilkan
equity curve yang indah dan sepenuhnya fiktif — dan itu pola yang dipakai
mayoritas proyek prediksi saham, termasuk prototipe di repo ini sebelum
di-refactor.

Yang dilakukan di sini: data dibagi menjadi blok berurutan. Untuk tiap blok,
model dilatih hanya memakai bar sebelum blok itu, lalu memprediksi blok itu.
Model dibangun ulang dari nol di tiap fold, sehingga tidak ada sisa keadaan
yang terbawa mundur.

Embargo
-------
Ada kebocoran halus yang tidak tertutup hanya dengan memisahkan latih dan uji.
Label ``y[t]`` menjawab apa yang terjadi di bar ``t + horizon``. Jadi baris
latih paling akhir sudah memuat informasi dari bar yang secara kronologis
berada di dalam periode uji. Jaraknya kecil, tidak akan memicu error apa pun,
dan cukup untuk membuat hasil terlihat lebih baik daripada yang sebenarnya.

``embargo`` membuang ``horizon`` baris terakhir dari tiap jendela latih.
Defaultnya 1, yang benar untuk prediksi satu bar ke depan. Kalau target dibuat
dengan ``horizon`` lebih besar, embargo harus dinaikkan sebanding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

__all__ = ["Fold", "WalkForwardResult", "walk_forward_signals"]


@dataclass(frozen=True)
class Fold:
    """Satu putaran latih-lalu-uji."""

    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    n_train: int
    fitted: bool
    """``False`` kalau fold dilewati karena data latihnya tidak memadai."""


@dataclass(frozen=True)
class WalkForwardResult:
    """Sinyal luar-sampel beserta catatan bagaimana sinyal itu dihasilkan."""

    signals: pd.Series
    """Bobot target, sejajar dengan index masukan. NaN di bar tanpa prediksi."""

    folds: tuple[Fold, ...]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def n_fitted(self) -> int:
        """Berapa kali model benar-benar dilatih ulang."""
        return sum(1 for fold in self.folds if fold.fitted)

    @property
    def first_prediction(self) -> Any | None:
        """Bar pertama yang punya sinyal; awal backtest yang sesungguhnya."""
        valid = self.signals.dropna()
        return None if valid.empty else valid.index[0]

    @property
    def coverage(self) -> float:
        """Bagian bar yang berhasil diprediksi."""
        if len(self.signals) == 0:
            return 0.0
        return float(self.signals.notna().mean())


def walk_forward_signals(
    factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_size: int = 252,
    test_size: int = 21,
    embargo: int = 1,
    expanding: bool = True,
    min_train_size: int | None = None,
) -> WalkForwardResult:
    """Hasilkan sinyal luar-sampel dengan melatih ulang secara berkala.

    Parameters
    ----------
    factory:
        Menghasilkan adapter baru tiap fold. Sengaja berupa factory, bukan satu
        objek model: model yang dipakai ulang antar fold membawa keadaan dari
        masa depan ke masa lalu.
    X, y:
        Fitur dan label, berbagi index yang sama. Baris dengan NaN dibuang dari
        pelatihan; bar yang fiturnya NaN tidak diprediksi dan tetap NaN, yang
        oleh engine diperlakukan sebagai posisi kas.
    train_size:
        Bar minimum sebelum prediksi pertama. Untuk data harian, 252 berarti
        model butuh sekitar setahun riwayat sebelum berpendapat.
    test_size:
        Panjang tiap blok uji, sekaligus jarak antar pelatihan ulang. Nilai
        kecil lebih realistis tapi jauh lebih lambat.
    embargo:
        Baris yang dibuang dari ujung jendela latih. Harus minimal sebesar
        horizon label. Lihat catatan di docstring modul.
    expanding:
        ``True`` memakai seluruh riwayat yang tersedia; ``False`` memakai
        jendela bergulir sepanjang ``train_size``, yang berguna kalau perilaku
        pasar diduga berubah.
    min_train_size:
        Baris bersih minimum agar fold layak dilatih. Default ``train_size//2``.
    """
    if not X.index.equals(y.index):
        raise ValueError("X dan y harus berbagi index yang sama")
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size dan test_size harus >= 1")
    if embargo < 0:
        raise ValueError("embargo tidak boleh negatif")

    n = len(X)
    minimum = train_size // 2 if min_train_size is None else min_train_size
    signals = pd.Series(np.nan, index=X.index, name="signal")

    feature_rows_valid = X.notna().all(axis=1)
    label_rows_valid = feature_rows_valid & y.notna()

    folds: list[Fold] = []
    start = train_size
    while start < n:
        stop = min(start + test_size, n)
        train_stop = start - embargo

        if train_stop <= 0:
            start = stop
            continue

        train_begin = 0 if expanding else max(0, train_stop - train_size)
        train_index = X.index[train_begin:train_stop]
        usable = label_rows_valid.loc[train_index]
        train_index = train_index[usable]

        test_index = X.index[start:stop]
        predictable = test_index[feature_rows_valid.loc[test_index]]

        fitted = False
        if len(train_index) >= minimum and len(predictable) > 0:
            y_train = y.loc[train_index]
            # Classifier butuh lebih dari satu kelas; pasar yang naik terus
            # sepanjang jendela latih bisa menghasilkan label seragam.
            if y_train.nunique() > 1:
                model = factory()
                model.fit(X.loc[train_index], y_train)
                signals.loc[predictable] = np.asarray(
                    model.predict_signal(X.loc[predictable]), dtype=float
                )
                fitted = True

        folds.append(
            Fold(
                train_start=train_index[0] if len(train_index) else None,
                train_end=train_index[-1] if len(train_index) else None,
                test_start=test_index[0],
                test_end=test_index[-1],
                n_train=len(train_index),
                fitted=fitted,
            )
        )
        start = stop

    return WalkForwardResult(signals=signals, folds=tuple(folds))
