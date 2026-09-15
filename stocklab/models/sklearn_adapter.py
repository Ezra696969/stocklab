"""Adapter untuk estimator scikit-learn.

Melayani dua tier sekaligus (PLAN.md sec. 1). Tier 1 membangun estimator dari
registry, jadi parameter dan rentangnya sudah diketahui. Tier 2 menerima
artifact terlatih milik user, dan di situ semuanya harus digali sendiri:

``feature_names_in_``
    Disimpan scikit-learn saat model dilatih dengan DataFrame. Ini yang
    memberi tahu aplikasi indikator apa yang harus dihitung, beserta urutannya
    — sehingga user tidak perlu menuliskan apa pun.
``get_params()``
    Hyperparameter berikut nilainya, langsung berubah jadi kontrol di panel.

Kalau model dilatih dengan array biasa dan bukan DataFrame, ``feature_names_in_``
tidak ada, dan tidak ada cara menebaknya. Saat itu terjadi user harus memilih
fiturnya sendiri; menebak urutan kolom orang lain adalah cara paling halus
untuk menghasilkan backtest yang salah tanpa pernah melempar error.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stocklab.models.base import (
    ModelSpec,
    PositionSizing,
    SignalKind,
    to_weights,
)
from stocklab.params import ParamSpec, infer_param_specs

__all__ = ["SklearnAdapter", "infer_signal_kind"]

#: Parameter yang tidak ada gunanya ditampilkan ke user.
HIDDEN_PARAMS = frozenset(
    {
        "n_jobs",
        "random_state",
        "verbose",
        "warm_start",
        "copy_X",
        "copy_X_train",
        "cache_size",
        "memory",
        "steps",
        "transformer_list",
        "estimator",
        "base_estimator",
        "estimators",
    }
)


def infer_signal_kind(estimator: Any) -> SignalKind:
    """Tebak bentuk keluaran dari kemampuan estimator.

    Ini tebakan yang aman karena hanya membaca metode yang benar-benar ada,
    bukan menerka maksud model.
    """
    if hasattr(estimator, "predict_proba"):
        return "proba"
    if getattr(estimator, "_estimator_type", None) == "classifier":
        return "class"
    if hasattr(estimator, "classes_"):
        return "class"
    return "expected_return"


class SklearnAdapter:
    """Bungkus estimator scikit-learn agar memenuhi :class:`ModelAdapter`.

    Parameters
    ----------
    estimator:
        Apa pun yang punya ``fit`` dan ``predict``; ``Pipeline`` juga bisa.
    name, label, description:
        Identitas untuk UI.
    signal_kind:
        Bentuk keluaran. Ditebak dari estimator kalau tidak disebut.
    param_specs:
        Spec parameter yang sudah disiapkan (Tier 1). Kalau ``None``, spec
        digali dari ``get_params()`` (Tier 2).
    feature_names:
        Fitur yang dibutuhkan. Kalau ``None``, diambil dari
        ``feature_names_in_`` bila estimator sudah terlatih.
    pretrained:
        ``True`` untuk artifact yang dipakai apa adanya tanpa dilatih ulang.
    train_end:
        Akhir data latih menurut user. Lihat :attr:`ModelSpec.train_end`.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        name: str = "sklearn",
        label: str | None = None,
        description: str = "",
        signal_kind: SignalKind | None = None,
        param_specs: Mapping[str, ParamSpec] | None = None,
        feature_names: Sequence[str] | None = None,
        pretrained: bool = False,
        train_end: pd.Timestamp | str | None = None,
        threshold: float = 0.5,
        allow_short: bool = False,
        sizing: PositionSizing = "binary",
    ):
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError(
                "estimator harus punya metode fit dan predict; "
                f"dapat {type(estimator).__name__}"
            )

        self.estimator = estimator
        self.name = name
        self.label = label or name.replace("_", " ").title()
        self.description = description
        self.signal_kind: SignalKind = signal_kind or infer_signal_kind(estimator)
        self.pretrained = pretrained
        self.threshold = threshold
        self.allow_short = allow_short
        self.sizing: PositionSizing = sizing
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None

        self._param_specs = dict(param_specs) if param_specs is not None else None
        self._feature_names = tuple(feature_names) if feature_names else None
        self._fitted = pretrained

    # -- kontrak ModelAdapter -------------------------------------------------

    def describe(self) -> ModelSpec:
        return ModelSpec(
            name=self.name,
            label=self.label,
            params=self._resolve_param_specs(),
            signal_kind=self.signal_kind,
            feature_names=self.feature_names,
            train_end=self.train_end,
            requires_fit=not self.pretrained,
            description=self.description,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.pretrained:
            raise NotImplementedError(
                f"'{self.name}' adalah model terlatih dan dipakai apa adanya; "
                "melatih ulang akan membuang bobot yang di-upload user"
            )
        X_aligned = self._align(X)
        self.estimator.fit(X_aligned, np.asarray(y))
        self._fitted = True

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError(
                f"'{self.name}' belum dilatih; panggil fit() atau pakai walk-forward"
            )
        X_aligned = self._align(X)
        raw = self._predict_raw(X_aligned)
        return to_weights(
            raw,
            self.signal_kind,
            index=X.index,
            threshold=self.threshold,
            allow_short=self.allow_short,
            sizing=self.sizing,
        )

    # -- internal -------------------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Fitur yang dibutuhkan model, dari konstruktor atau dari artifact."""
        if self._feature_names is not None:
            return self._feature_names
        found = getattr(self.estimator, "feature_names_in_", None)
        if found is None:
            # Pipeline menyimpannya di langkah pertama.
            steps = getattr(self.estimator, "steps", None)
            if steps:
                found = getattr(steps[0][1], "feature_names_in_", None)
        return tuple(str(name) for name in found) if found is not None else None

    def _resolve_param_specs(self) -> dict[str, ParamSpec]:
        if self._param_specs is not None:
            return dict(self._param_specs)
        try:
            params = self.estimator.get_params(deep=False)
        except (AttributeError, TypeError):
            return {}
        # Nilai yang bukan skalar (estimator bersarang, callable) otomatis jadi
        # opaque dan tidak dirender; infer_param_specs yang memutuskan.
        return infer_param_specs(params, skip=HIDDEN_PARAMS)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pastikan kolom sesuai yang diharapkan model, dalam urutan yang benar."""
        names = self.feature_names
        if names is None:
            return X
        missing = [name for name in names if name not in X.columns]
        if missing:
            raise ValueError(
                f"'{self.name}' butuh fitur {missing} yang tidak ada; "
                f"tersedia: {list(X.columns)}"
            )
        return X[list(names)]

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        if self.signal_kind == "proba":
            proba = self.estimator.predict_proba(X)
            return self._positive_class_column(proba)
        return np.asarray(self.estimator.predict(X), dtype=float)

    def _positive_class_column(self, proba: np.ndarray) -> np.ndarray:
        """Ambil kolom peluang untuk kelas 'naik'.

        Tidak bisa berasumsi kolom terakhir: urutan kelas ditentukan
        ``classes_``, dan model yang dilatih pada label ``{-1, 1}`` atau
        ``{0, 1}`` menaruhnya di tempat berbeda.
        """
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            return proba
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            return proba[:, -1]
        classes = np.asarray(classes)
        positive = classes.max()
        column = int(np.flatnonzero(classes == positive)[0])
        return proba[:, column]
