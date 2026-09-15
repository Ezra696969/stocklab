"""Model zoo bawaan (Tier 1).

Jalur masuk untuk user yang belum punya model apa pun: pilih dari daftar,
geser beberapa slider, jalankan. Tidak ada file yang perlu di-upload dan tidak
ada kode yang perlu ditulis.

Berbeda dengan model yang di-upload, di sini rentang parameter benar-benar
diketahui — model-modelnya kita sendiri yang pilih dan defaultnya kita sendiri
yang tentukan — jadi :data:`PARAM_HINTS` mengisi batas yang layak dan UI boleh
memakai slider. Untuk artifact milik user, jalurnya lewat introspeksi di
:mod:`stocklab.models.sklearn_adapter`, dan parameter yang batasnya tidak
diketahui dirender sebagai kotak angka.

Setiap model dibungkus ``StandardScaler`` lewat ``Pipeline``. Tanpa itu, fitur
berskala harga (SMA di angka ribuan) akan menenggelamkan fitur berskala kecil
(RSI 0-100) pada model berbasis jarak dan regularisasi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from stocklab.models.base import PositionSizing, SignalKind
from stocklab.models.sklearn_adapter import SklearnAdapter
from stocklab.params import ParamSpec, infer_param_specs

__all__ = ["MODELS", "ModelEntry", "PARAM_HINTS", "build_model", "get_model_entry"]


#: Rentang untuk hyperparameter yang lazim. Dikurasi tangan, bukan ditebak.
PARAM_HINTS: dict[str, ParamSpec] = {
    "n_estimators": ParamSpec(
        "n_estimators", "int", 300, 10, 1000, 10,
        label="Jumlah pohon",
        help="Lebih banyak pohon menstabilkan prediksi tapi memperlambat pelatihan.",
    ),
    "max_depth": ParamSpec(
        "max_depth", "int", 8, 1, 32, 1,
        label="Kedalaman maksimum",
        help="Pohon yang dalam menangkap pola rumit sekaligus lebih mudah overfit.",
    ),
    "min_samples_leaf": ParamSpec(
        "min_samples_leaf", "int", 20, 1, 200, 1,
        label="Minimum sampel per daun",
        help="Menaikkannya memaksa model menemukan pola yang berulang, bukan kebetulan.",
    ),
    "learning_rate": ParamSpec(
        "learning_rate", "float", 0.05, 0.001, 1.0, 0.001,
        label="Laju belajar",
        help="Langkah kecil biasanya menghasilkan model yang lebih baik, asal pohonnya cukup.",
    ),
    "C": ParamSpec(
        "C", "float", 1.0, 0.001, 100.0, 0.001,
        label="Kekuatan regularisasi (C)",
        help="Nilai kecil berarti regularisasi kuat dan model lebih sederhana.",
    ),
    "alpha": ParamSpec(
        "alpha", "float", 1.0, 0.0001, 100.0, 0.0001,
        label="Alpha",
        help="Besar penalti terhadap koefisien yang membesar.",
    ),
    "max_iter": ParamSpec(
        "max_iter", "int", 1000, 100, 10_000, 100,
        label="Iterasi maksimum",
        help="Batas atas proses optimasi.",
    ),
    "max_features": ParamSpec(
        "max_features", "choice", "sqrt",
        choices=("sqrt", "log2", None),
        label="Fitur per split",
    ),
    "subsample": ParamSpec(
        "subsample", "float", 1.0, 0.3, 1.0, 0.05,
        label="Subsample",
        help="Bagian data yang dipakai tiap tahap; di bawah 1.0 menambah keragaman.",
    ),
}


@dataclass(frozen=True)
class ModelEntry:
    """Satu model bawaan beserta default dan rentang parameternya."""

    name: str
    label: str
    description: str
    factory: Callable[..., Any]
    """Membangun estimator dari parameter yang sudah divalidasi."""

    defaults: dict[str, Any]
    signal_kind: SignalKind = "proba"
    hints: dict[str, ParamSpec] = field(default_factory=dict)

    def param_specs(self) -> dict[str, ParamSpec]:
        """Spec parameter, dengan hint menimpa tebakan tipe.

        Default diambil dari entri ini, bukan dari default scikit-learn, supaya
        nilai yang ditampilkan panel benar-benar nilai yang dipakai.
        """
        overrides = {}
        for name, default in self.defaults.items():
            hint = self.hints.get(name) or PARAM_HINTS.get(name)
            if hint is not None:
                overrides[name] = ParamSpec(
                    name=name,
                    kind=hint.kind,
                    default=default,
                    low=hint.low,
                    high=hint.high,
                    step=hint.step,
                    choices=hint.choices,
                    label=hint.label,
                    help=hint.help,
                )
        return infer_param_specs(self.defaults, overrides=overrides)

    def coerce(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validasi parameter user terhadap spec, isi sisanya dengan default."""
        specs = self.param_specs()
        unknown = set(params or {}) - set(specs)
        if unknown:
            raise ValueError(
                f"parameter tidak dikenal untuk '{self.name}': {sorted(unknown)}; "
                f"yang tersedia: {sorted(specs)}"
            )
        resolved = dict(self.defaults)
        for name, value in (params or {}).items():
            resolved[name] = specs[name].coerce(value)
        return resolved


def _pipeline(estimator: Any) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _logistic(**params: Any) -> Any:
    from sklearn.linear_model import LogisticRegression

    return _pipeline(LogisticRegression(random_state=0, **params))


def _random_forest(**params: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return _pipeline(RandomForestClassifier(random_state=0, n_jobs=-1, **params))


def _gradient_boosting(**params: Any) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    return _pipeline(GradientBoostingClassifier(random_state=0, **params))


def _ridge(**params: Any) -> Any:
    from sklearn.linear_model import Ridge

    return _pipeline(Ridge(random_state=0, **params))


MODELS: dict[str, ModelEntry] = {
    "logistic_regression": ModelEntry(
        name="logistic_regression",
        label="Regresi Logistik",
        description=(
            "Model linier yang memperkirakan peluang harga naik. Cepat, mudah "
            "dibaca, dan menjadi pembanding yang jujur: model rumit yang tidak "
            "bisa mengalahkannya biasanya cuma menghafal derau."
        ),
        factory=_logistic,
        defaults={"C": 1.0, "max_iter": 1000},
        signal_kind="proba",
    ),
    "random_forest": ModelEntry(
        name="random_forest",
        label="Random Forest",
        description=(
            "Kumpulan pohon keputusan yang menangkap hubungan tidak linier "
            "antar indikator. Tahan terhadap fitur berskala beda dan relatif "
            "sulit dibuat overfit parah."
        ),
        factory=_random_forest,
        defaults={
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
        },
        signal_kind="proba",
    ),
    "gradient_boosting": ModelEntry(
        name="gradient_boosting",
        label="Gradient Boosting",
        description=(
            "Pohon yang dibangun berurutan, masing-masing memperbaiki kesalahan "
            "pendahulunya. Biasanya paling akurat di data tabular, tapi paling "
            "gampang overfit kalau lajunya terlalu besar."
        ),
        factory=_gradient_boosting,
        defaults={
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 1.0,
        },
        signal_kind="proba",
    ),
    "ridge": ModelEntry(
        name="ridge",
        label="Regresi Ridge",
        description=(
            "Memperkirakan besar return berikutnya, bukan sekadar arahnya. "
            "Cocok dipadukan dengan sizing proporsional supaya prediksi yang "
            "ragu-ragu tidak mendapat modal sebesar prediksi yang yakin."
        ),
        factory=_ridge,
        defaults={"alpha": 1.0},
        signal_kind="expected_return",
    ),
}


def get_model_entry(name: str) -> ModelEntry:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"model '{name}' tidak dikenal; yang tersedia: {sorted(MODELS)}"
        ) from None


def build_model(
    name: str,
    params: Mapping[str, Any] | None = None,
    *,
    threshold: float = 0.5,
    allow_short: bool = False,
    sizing: PositionSizing = "binary",
) -> SklearnAdapter:
    """Bangun adapter siap latih dari model zoo.

    Parameters
    ----------
    name:
        Kunci di :data:`MODELS`.
    params:
        Parameter dari panel UI. Yang tidak disebut memakai default entri.
    threshold, allow_short, sizing:
        Aturan sizing, diteruskan ke adapter. Lihat
        :func:`stocklab.models.base.to_weights`.
    """
    entry = get_model_entry(name)
    resolved = entry.coerce(params)

    return SklearnAdapter(
        entry.factory(**resolved),
        name=entry.name,
        label=entry.label,
        description=entry.description,
        signal_kind=entry.signal_kind,
        param_specs=entry.param_specs(),
        threshold=threshold,
        allow_short=allow_short,
        sizing=sizing,
    )
