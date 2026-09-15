"""Kontrak model dan konversi keluaran mentah menjadi bobot.

Engine backtest hanya menerima bobot target. Modul inilah perbatasannya: di
sisi sini ada probabilitas, kelas, dan expected return dengan satuan
masing-masing; di sisi sana hanya ada angka antara -1 dan 1 yang berarti berapa
bagian modal yang dipegang. Semua penerjemahan terjadi di sini, sekali, supaya
tidak ada model yang diam-diam punya aturan sizing sendiri.

Pembagian ini juga yang membuat Tier 3 nanti cukup menambah satu adapter:
selama model bisa menghasilkan salah satu dari tiga bentuk keluaran di atas,
sisa sistem tidak perlu tahu apa isinya.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from stocklab.params import ParamSpec

__all__ = [
    "ModelAdapter",
    "ModelSpec",
    "PositionSizing",
    "SignalKind",
    "to_weights",
]

SignalKind = Literal["proba", "class", "expected_return"]
"""Bentuk keluaran mentah sebuah model.

``proba``
    Peluang harga naik, antara 0 dan 1.
``class``
    Keputusan langsung: 1 untuk naik, 0 atau -1 untuk turun.
``expected_return``
    Perkiraan besar return; yang dipakai hanya arahnya kecuali sizing diminta
    proporsional.
"""

PositionSizing = Literal["binary", "scaled"]
"""``binary`` masuk penuh atau tidak sama sekali; ``scaled`` menyesuaikan
besar posisi dengan keyakinan model."""


@dataclass(frozen=True)
class ModelSpec:
    """Deskripsi diri sebuah model, cukup untuk merender panel parameter.

    Inilah yang dibaca panel kanan-bawah. Model bawaan mengisinya dari entri
    registry; model yang di-upload mengisinya lewat introspeksi artifact.
    """

    name: str
    label: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    signal_kind: SignalKind = "proba"
    feature_names: tuple[str, ...] | None = None
    """Fitur yang dibutuhkan model, beserta urutannya.

    ``None`` berarti tidak diketahui — lazim untuk model yang belum dilatih.
    Untuk artifact yang di-upload, inilah yang memberi tahu aplikasi indikator
    apa yang harus dihitung.
    """

    train_end: pd.Timestamp | None = None
    """Tanggal akhir data latih, kalau user menyebutkannya.

    Untuk model yang dilatih di dalam aplikasi, ini diisi otomatis oleh
    walk-forward. Untuk artifact yang di-upload, aplikasi tidak punya cara
    mengetahuinya: model bisa saja sudah melihat periode yang akan di-backtest,
    dan tidak ada kode yang bisa mendeteksi itu. Lihat PLAN.md sec. 6.
    """

    requires_fit: bool = True
    """``False`` untuk model yang sudah terlatih dan dipakai apa adanya."""

    description: str = ""

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.params.items()}

    @property
    def is_pretrained(self) -> bool:
        return not self.requires_fit

    @property
    def leakage_is_unverifiable(self) -> bool:
        """Model terlatih tanpa ``train_end`` tidak bisa dijamin bersih.

        UI memakai ini untuk memunculkan peringatan. Hasilnya tetap dihitung —
        keputusan ada di user — tapi user harus tahu apa yang sedang dilihatnya.
        """
        return self.is_pretrained and self.train_end is None


@runtime_checkable
class ModelAdapter(Protocol):
    """Yang harus dipenuhi apa pun agar bisa di-backtest di sini.

    Tiga metode, dan tidak satu pun menyebut scikit-learn, torch, atau format
    file. Model bawaan, artifact yang di-upload, dan model time-series dari
    Hugging Face nanti sama-sama masuk lewat pintu ini.
    """

    def describe(self) -> ModelSpec:
        """Kembalikan spec model, termasuk parameter yang bisa diatur."""
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Latih model. Model pretrained boleh melempar ``NotImplementedError``."""
        ...

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        """Hasilkan bobot target untuk tiap baris ``X``.

        Keluarannya sudah dalam skala yang dipahami engine: antara -1 dan 1,
        sejajar dengan ``X.index``.
        """
        ...


def to_weights(
    raw: pd.Series | np.ndarray,
    kind: SignalKind,
    *,
    index: pd.Index | None = None,
    threshold: float = 0.5,
    allow_short: bool = False,
    sizing: PositionSizing = "binary",
) -> pd.Series:
    """Ubah keluaran mentah model menjadi bobot target.

    Parameters
    ----------
    raw:
        Keluaran model, dalam satuan yang ditentukan ``kind``.
    kind:
        Bentuk keluaran. Lihat :data:`SignalKind`.
    threshold:
        Untuk ``proba``, batas keyakinan sebelum posisi dibuka. Menaikkannya
        membuat model lebih pemilih dan menurunkan turnover.
    allow_short:
        Kalau ``False``, bobot negatif dipangkas ke nol. Default konservatif:
        short tidak selalu tersedia di semua bursa, dan biaya pinjamnya tidak
        dimodelkan di sini.
    sizing:
        ``binary`` masuk penuh atau tidak sama sekali. ``scaled`` membuat besar
        posisi mengikuti keyakinan model, sehingga prediksi yang ragu-ragu tidak
        mendapat modal sebesar prediksi yang yakin.
    """
    if isinstance(raw, pd.Series):
        index = raw.index if index is None else index
        values = raw.to_numpy(dtype=float)
    else:
        values = np.asarray(raw, dtype=float)
        if index is None:
            index = pd.RangeIndex(len(values))

    if len(values) != len(index):
        raise ValueError(
            f"keluaran model panjangnya {len(values)}, index {len(index)}"
        )
    if not 0.0 < threshold < 1.0 and kind == "proba":
        raise ValueError(f"threshold harus di antara 0 dan 1, dapat {threshold}")

    missing = np.isnan(values)

    if kind == "proba":
        weights = _weights_from_proba(values, threshold, allow_short, sizing)
    elif kind == "class":
        weights = np.sign(values) if allow_short else (values > 0).astype(float)
    elif kind == "expected_return":
        weights = _weights_from_expected_return(values, allow_short, sizing)
    else:
        raise ValueError(f"signal_kind tidak dikenal: {kind!r}")

    lower = -1.0 if allow_short else 0.0
    weights = np.clip(weights, lower, 1.0)
    # NaN tetap NaN: engine memperlakukannya sebagai posisi kas, dan itu jawaban
    # yang benar untuk bar yang modelnya belum bisa berpendapat.
    weights[missing] = np.nan
    return pd.Series(weights, index=index, name="weight")


def _weights_from_proba(
    values: np.ndarray, threshold: float, allow_short: bool, sizing: PositionSizing
) -> np.ndarray:
    if sizing == "binary":
        weights = (values >= threshold).astype(float)
        if allow_short:
            weights = np.where(values <= 1.0 - threshold, -1.0, weights)
        return weights
    if sizing == "scaled":
        if allow_short:
            # p=0.5 berarti tidak punya pendapat, jadi tidak ada posisi.
            return (values - 0.5) * 2.0
        return (values - threshold) / (1.0 - threshold)
    raise ValueError(f"sizing tidak dikenal: {sizing!r}")


def _weights_from_expected_return(
    values: np.ndarray, allow_short: bool, sizing: PositionSizing
) -> np.ndarray:
    if sizing == "binary":
        return np.sign(values) if allow_short else (values > 0).astype(float)
    # Skala keyakinan relatif terhadap sebaran prediksi model itu sendiri;
    # expected return tidak punya satuan yang bisa dipakai apa adanya sebagai
    # bobot. tanh menjaga hasilnya tetap di dalam batas tanpa memotong keras.
    finite = values[np.isfinite(values)]
    scale = np.std(finite) if len(finite) > 1 else 0.0
    if scale == 0:
        return np.sign(values) if allow_short else (values > 0).astype(float)
    scaled = np.tanh(values / (2.0 * scale))
    return scaled if allow_short else np.clip(scaled, 0.0, 1.0)
