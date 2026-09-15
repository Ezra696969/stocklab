"""Keadaan sesi, termasuk pembukuan percobaan.

Bagian terpenting di modul ini adalah :class:`TrialLog`. Tiap kali user menekan
Jalankan — menggeser periode RSI, mengganti model, menaikkan jumlah pohon — itu
satu percobaan tambahan dalam pencarian. Geser cukup sering, dan konfigurasi
terbaik akan terlihat meyakinkan tanpa alasan apa pun.

Aplikasi backtesting biasanya tidak menghitungnya. Yang ditampilkan hanya hasil
run terakhir, seolah-olah itu satu-satunya yang pernah dicoba, padahal user
sudah menjalankan empat puluh sebelumnya. Log ini mencatat semuanya dan
mengumpankannya ke deflated Sharpe, sehingga angka yang ditampilkan mengetahui
berapa kali pencarian benar-benar dilakukan.

Dibuat sebagai kelas biasa tanpa menyentuh Streamlit, supaya bisa diuji sendiri.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import numpy as np

__all__ = ["Trial", "TrialLog", "session_defaults"]


@dataclass(frozen=True)
class Trial:
    """Satu run yang pernah dijalankan user dalam sesi ini."""

    model: str
    params: Mapping[str, Any]
    features: tuple[str, ...]
    sharpe: float
    equity_ratio: float
    at: datetime = field(default_factory=datetime.now)

    def signature(self) -> tuple:
        """Identitas konfigurasi, untuk mengenali run yang persis sama.

        Menjalankan ulang konfigurasi yang sama bukan percobaan baru: tidak ada
        pencarian yang bertambah kalau tidak ada yang berubah.
        """
        return (
            self.model,
            tuple(sorted((str(k), str(v)) for k, v in self.params.items())),
            self.features,
        )


class TrialLog:
    """Catatan seluruh konfigurasi yang dicoba dalam satu sesi."""

    def __init__(self) -> None:
        self._trials: list[Trial] = []

    def __len__(self) -> int:
        return len(self._trials)

    def __iter__(self):
        return iter(self._trials)

    @property
    def trials(self) -> tuple[Trial, ...]:
        return tuple(self._trials)

    def record(self, trial: Trial) -> None:
        self._trials.append(trial)

    @property
    def n_distinct(self) -> int:
        """Konfigurasi berbeda yang pernah dicoba.

        Inilah angka yang dipakai koreksi multiple testing. Menekan Jalankan dua
        kali pada pengaturan yang sama tidak menambah pencarian.
        """
        return len({trial.signature() for trial in self._trials})

    def sharpes(self) -> np.ndarray:
        """Sharpe dari tiap konfigurasi berbeda, untuk deflated Sharpe.

        Sebarannya lebih informatif daripada sekadar jumlahnya: pencarian yang
        hasilnya berserakan lebar lebih gampang melahirkan pemenang kebetulan.
        """
        best: dict[tuple, float] = {}
        for trial in self._trials:
            if not np.isnan(trial.sharpe):
                best[trial.signature()] = trial.sharpe
        return np.array(list(best.values()), dtype=float)

    def best(self) -> Trial | None:
        usable = [t for t in self._trials if not np.isnan(t.sharpe)]
        return max(usable, key=lambda t: t.sharpe) if usable else None

    def summary(self) -> dict[str, float]:
        values = self.sharpes()
        return {
            "n_runs": float(len(self._trials)),
            "n_distinct": float(self.n_distinct),
            "sharpe_spread": float(values.max() - values.min()) if len(values) > 1 else 0.0,
            "best_sharpe": float(values.max()) if len(values) else float("nan"),
        }

    def clear(self) -> None:
        self._trials.clear()


def session_defaults() -> dict[str, Any]:
    """Nilai awal ``st.session_state``.

    Dikumpulkan di satu tempat supaya panel tidak saling menebak kunci apa yang
    sudah ada dan kapan.
    """
    return {
        "trials": TrialLog(),
        "messages": [],
        "result": None,
        "walkforward": None,
        "validation": None,
        "ohlcv": None,
        "features": None,
        "model_spec": None,
        "uploaded_adapter": None,
    }
