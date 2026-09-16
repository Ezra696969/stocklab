"""Tier 3 — model peramal deret waktu dari Hugging Face.

Berbeda jenis dari dua tier sebelumnya. Model di Tier 1 dan 2 adalah classifier
tabular: diberi sebaris indikator, mereka menjawab naik atau turun. Model di
sini adalah *foundation model* deret waktu — diberi potongan riwayat harga,
mereka meramalkan kelanjutannya, dan tidak pernah dilatih pada saham sama
sekali. Kemampuan zero-shot itulah yang membuatnya menarik untuk diuji.

Dari ramalan ke bobot
---------------------
Ramalannya berupa sebaran, bukan satu angka, dan itu justru menguntungkan.
Peluang harga naik dibaca dari posisi harga terakhir pada kurva kuantil ramalan:
kalau harga sekarang jatuh di kuantil bawah sebaran, sebagian besar kemungkinan
masa depan berada di atasnya. Hasilnya langsung berupa probabilitas, sehingga
masuk ke ``signal_kind="proba"`` yang sama dengan model Tier 1 dan memakai
ambang serta sizing yang sama persis.

Yang dipakai ``predict_quantiles``, bukan ``predict``, karena hanya itu yang
punya bentuk keluaran sama di kedua keluarga model: Chronos T5 menghasilkan
sampel, Chronos-Bolt menghasilkan kuantil langsung.

Kausalitas
----------
Ramalan untuk bar ``t`` hanya memakai bar sebelum ``t``, dan hasilnya tetap
melewati ``execution_lag`` di engine seperti sinyal lain. Model ini tidak dilatih
di sini, jadi tidak ada walk-forward — tapi juga tidak ada kebocoran dari
pelatihan, karena bobotnya dibuat tanpa pernah melihat data ini.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocklab.models.base import ModelSpec, PositionSizing, to_weights
from stocklab.params import ParamSpec

__all__ = [
    "FORECASTERS",
    "ForecastAdapter",
    "ForecasterEntry",
    "build_forecaster",
    "get_forecaster_entry",
    "resolve_device",
]

#: Kolom fitur yang dibaca peramal. Berbeda dengan model tabular yang menerima
#: banyak indikator, peramal hanya butuh satu deret: harganya sendiri.
PRICE_COLUMN = "price_log"

#: Kuantil yang diminta. Cukup rapat untuk menaksir peluang dengan interpolasi,
#: tanpa memperlambat inferensi secara berarti.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class ForecasterEntry:
    """Satu model peramal yang dikurasi."""

    name: str
    model_id: str
    label: str
    description: str
    params_mb: int
    """Ukuran unduhan kira-kira, supaya user tahu apa yang akan terjadi."""

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            "context_length": ParamSpec(
                "context_length", "int", 128, 32, 512, 16,
                label="Panjang konteks",
                help=(
                    "Berapa bar riwayat yang dilihat model tiap kali meramal. "
                    "Lebih panjang memberi lebih banyak pola, tapi lebih lambat."
                ),
            ),
            "prediction_length": ParamSpec(
                "prediction_length", "int", 1, 1, 5, 1,
                label="Horizon ramalan",
                help="Berapa bar ke depan yang diramalkan sebelum diubah jadi sinyal.",
            ),
            "batch_size": ParamSpec(
                "batch_size", "int", 128, 8, 512, 8,
                label="Ukuran batch",
                help=(
                    "Berapa ramalan dihitung sekaligus. Batch besar jauh lebih "
                    "cepat tapi memakan lebih banyak memori."
                ),
            ),
        }

    def coerce(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        specs = self.param_specs()
        unknown = set(params or {}) - set(specs)
        if unknown:
            raise ValueError(
                f"parameter tidak dikenal untuk '{self.name}': {sorted(unknown)}"
            )
        resolved = {name: spec.default for name, spec in specs.items()}
        for name, value in (params or {}).items():
            resolved[name] = specs[name].coerce(value)
        return resolved


FORECASTERS: dict[str, ForecasterEntry] = {
    "chronos_bolt_tiny": ForecasterEntry(
        name="chronos_bolt_tiny",
        model_id="amazon/chronos-bolt-tiny",
        label="Chronos-Bolt Tiny",
        description=(
            "Peramal deret waktu Amazon, varian tercepat. Tidak pernah dilatih "
            "pada data saham — ramalannya sepenuhnya zero-shot, jadi tidak ada "
            "kebocoran dari pelatihan."
        ),
        params_mb=9,
    ),
    "chronos_bolt_small": ForecasterEntry(
        name="chronos_bolt_small",
        model_id="amazon/chronos-bolt-small",
        label="Chronos-Bolt Small",
        description=(
            "Chronos-Bolt yang lebih besar. Ramalannya biasanya lebih halus, "
            "dengan waktu inferensi beberapa kali lipat."
        ),
        params_mb=48,
    ),
    "chronos_t5_tiny": ForecasterEntry(
        name="chronos_t5_tiny",
        model_id="amazon/chronos-t5-tiny",
        label="Chronos T5 Tiny",
        description=(
            "Chronos generasi pertama, berbasis T5. Menghasilkan sebaran lewat "
            "pengambilan sampel, bukan kuantil langsung."
        ),
        params_mb=32,
    ),
    "chronos_t5_small": ForecasterEntry(
        name="chronos_t5_small",
        model_id="amazon/chronos-t5-small",
        label="Chronos T5 Small",
        description="Chronos T5 ukuran menengah; paling lambat dari daftar ini.",
        params_mb=185,
    ),
}


def get_forecaster_entry(name: str) -> ForecasterEntry:
    try:
        return FORECASTERS[name]
    except KeyError:
        raise KeyError(
            f"peramal '{name}' tidak dikenal; yang tersedia: {sorted(FORECASTERS)}"
        ) from None


def resolve_device(preferred: str = "auto") -> str:
    """Pilih device, dengan membuktikan bukan dengan mempercayai flag.

    ``torch.cuda.is_available()`` hanya berarti ada driver dan perangkat yang
    terlihat. Perangkat itu bisa saja punya compute capability yang tidak
    didukung build torch terpasang, dan kegagalannya baru muncul jauh di dalam
    inferensi sebagai "no kernel image is available for execution on the
    device" — persis saat backtest sudah berjalan lama.

    Jadi kandidatnya diuji dengan satu operasi sungguhan lebih dulu. Satu
    perkalian matriks kecil jauh lebih murah daripada kegagalan di tengah jalan.
    """
    import torch

    if preferred != "auto":
        return preferred
    if not torch.cuda.is_available():
        return "cpu"
    try:
        probe = torch.zeros(8, 8, device="cuda")
        torch.mm(probe, probe).cpu()
    except Exception:
        return "cpu"
    return "cuda"


def load_pipeline(model_id: str, device: str = "auto") -> Any:
    """Muat pipeline Chronos, dengan cache antar pemanggilan.

    Pemuatan pertama mengunduh bobot dan bisa memakan puluhan detik; sesudahnya
    berkasnya ada di cache Hugging Face dan pemuatan jauh lebih cepat. Cache di
    sini menghindari pemuatan ulang dalam satu sesi.
    """
    import torch
    from chronos import BaseChronosPipeline

    device = resolve_device(device)

    key = (model_id, device)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )
    return _PIPELINE_CACHE[key]


class ForecastAdapter:
    """Bungkus peramal deret waktu agar memenuhi :class:`ModelAdapter`."""

    def __init__(
        self,
        entry: ForecasterEntry,
        *,
        context_length: int = 128,
        prediction_length: int = 1,
        batch_size: int = 128,
        device: str = "auto",
        price_column: str = PRICE_COLUMN,
        threshold: float = 0.5,
        allow_short: bool = False,
        sizing: PositionSizing = "binary",
        pipeline: Any | None = None,
    ):
        self.entry = entry
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.price_column = price_column
        self.threshold = threshold
        self.allow_short = allow_short
        self.sizing: PositionSizing = sizing
        self.signal_kind = "proba"
        self._pipeline = pipeline

    # -- kontrak ModelAdapter -------------------------------------------------

    def describe(self) -> ModelSpec:
        return ModelSpec(
            name=self.entry.name,
            label=self.entry.label,
            params=self.entry.param_specs(),
            signal_kind="proba",
            feature_names=(self.price_column,),
            train_end=None,
            requires_fit=False,
            description=(
                f"{self.entry.description} Unduhan sekitar "
                f"{self.entry.params_mb} MB."
            ),
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError(
            f"'{self.entry.label}' dipakai zero-shot; melatihnya di sini justru "
            "menghapus keunggulannya, yaitu bobot yang belum pernah melihat data ini"
        )

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        if self.price_column not in X.columns:
            raise ValueError(
                f"peramal butuh kolom '{self.price_column}'; tersedia: "
                f"{list(X.columns)}. Tambahkan indikator Harga di panel Indikator."
            )

        series = X[self.price_column].to_numpy(dtype=float)
        probabilities = self._forecast(series)
        return to_weights(
            probabilities,
            "proba",
            index=X.index,
            threshold=self.threshold,
            allow_short=self.allow_short,
            sizing=self.sizing,
        )

    # -- internal -------------------------------------------------------------

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = load_pipeline(self.entry.model_id, self.device)
        return self._pipeline

    def _forecast(self, series: np.ndarray) -> np.ndarray:
        """Hasilkan peluang naik untuk tiap bar.

        Bar yang riwayatnya belum cukup panjang, atau yang konteksnya memuat NaN,
        dikembalikan sebagai NaN — sama seperti warmup indikator, dan engine
        memperlakukannya sebagai posisi kas.
        """
        import torch

        n = len(series)
        output = np.full(n, np.nan, dtype=float)

        positions, contexts, anchors = [], [], []
        for index in range(self.context_length, n):
            window = series[index - self.context_length : index]
            if np.isnan(window).any():
                continue
            positions.append(index)
            contexts.append(torch.tensor(window, dtype=torch.float32))
            anchors.append(window[-1])

        if not contexts:
            return output

        levels = list(QUANTILE_LEVELS)
        probabilities: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start : start + self.batch_size]
            quantiles, _ = self.pipeline.predict_quantiles(
                batch,
                prediction_length=self.prediction_length,
                quantile_levels=levels,
            )
            # (batch, horizon, kuantil) -> ambil langkah terakhir yang diramalkan
            final = np.asarray(quantiles.float().cpu().numpy())[:, -1, :]
            for row, anchor in zip(final, anchors[start : start + len(batch)]):
                probabilities.append(_probability_above(anchor, row, levels))

        output[positions] = probabilities
        return output


def _probability_above(
    anchor: float, quantiles: np.ndarray, levels: list[float]
) -> float:
    """Peluang nilai masa depan melampaui ``anchor``, dari kurva kuantilnya.

    Kurva kuantil adalah fungsi distribusi kumulatif yang dibalik. Mencari di
    kuantil berapa ``anchor`` berada memberi ``P(masa depan <= anchor)``, dan
    sisanya adalah peluang naik.

    ``np.interp`` menjepit di kedua ujung, yang justru perilaku yang diinginkan:
    harga di bawah seluruh kuantil ramalan berarti peluang naiknya mendekati
    satu, bukan tak terdefinisi.
    """
    ordered = np.maximum.accumulate(quantiles)  # jaga monotonisitas
    if ordered[-1] <= ordered[0]:
        # Ramalan tanpa sebaran sama sekali; tidak ada dasar untuk berpendapat.
        return 0.5
    below = float(np.interp(anchor, ordered, levels))
    return float(np.clip(1.0 - below, 0.0, 1.0))


def build_forecaster(
    name: str,
    params: Mapping[str, Any] | None = None,
    *,
    device: str = "auto",
    threshold: float = 0.5,
    allow_short: bool = False,
    sizing: PositionSizing = "binary",
    pipeline: Any | None = None,
) -> ForecastAdapter:
    """Bangun adapter peramal dari daftar kurasi."""
    entry = get_forecaster_entry(name)
    resolved = entry.coerce(params)
    return ForecastAdapter(
        entry,
        device=device,
        threshold=threshold,
        allow_short=allow_short,
        sizing=sizing,
        pipeline=pipeline,
        **resolved,
    )
