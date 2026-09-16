"""Adapter untuk artifact pickle, dijalankan di proses terpisah.

Artifact pretrained tidak pernah dilatih ulang, jadi :meth:`predict_signal`
dipanggil satu kali untuk seluruh rentang. Itu membuat proses sekali-pakai jadi
pilihan yang tepat: tidak perlu menjaga proses tetap hidup, tidak perlu protokol
IPC, dan tiap panggilan mulai dari keadaan bersih.

Baca :mod:`stocklab.models.worker` untuk apa yang sebenarnya diberikan dan tidak
diberikan pemisahan proses ini. Ringkasnya: ia membatasi kecelakaan, bukan
serangan.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stocklab.models.artifact import LoadPolicy
from stocklab.models.base import ModelSpec, PositionSizing, SignalKind, to_weights
from stocklab.params import ParamSpec, infer_param_specs
from stocklab.models.sklearn_adapter import HIDDEN_PARAMS

__all__ = ["SandboxError", "SandboxedAdapter", "run_worker"]


class SandboxError(RuntimeError):
    """Proses pekerja gagal, menggantung, atau menjawab dengan error."""


def run_worker(
    artifact: Path, command: str, policy: LoadPolicy, **payload: Any
) -> Any:
    """Jalankan satu perintah di proses terpisah dan kembalikan hasilnya.

    Proses dibunuh kalau melewati ``policy.timeout_seconds``. Itu satu-satunya
    batas yang bisa ditegakkan di Windows — modul ``resource`` untuk membatasi
    memori tidak tersedia di sana.
    """
    with tempfile.TemporaryDirectory(prefix="stocklab-worker-") as workspace:
        folder = Path(workspace)
        request_path = folder / "request.pkl"
        response_path = folder / "response.pkl"

        with request_path.open("wb") as handle:
            pickle.dump(
                {"artifact": str(artifact), "command": command, **payload},
                handle,
                protocol=4,
            )

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "stocklab.models.worker",
                 str(request_path), str(response_path)],
                capture_output=True,
                timeout=policy.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise SandboxError(
                f"artifact tidak selesai diproses dalam "
                f"{policy.timeout_seconds:.0f} detik dan prosesnya dihentikan"
            ) from None

        if not response_path.exists():
            detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
            raise SandboxError(
                "proses pemuat berhenti tanpa menghasilkan jawaban"
                + (f": {detail[-500:]}" if detail else "")
            )

        with response_path.open("rb") as handle:
            response = pickle.load(handle)

    if not response.get("ok"):
        raise SandboxError(str(response.get("error", "kegagalan tanpa keterangan")))
    return response["result"]


class SandboxedAdapter:
    """Artifact scikit-learn yang dimuat di proses terpisah.

    Parameters
    ----------
    path:
        Berkas yang sudah lolos :func:`~stocklab.models.artifact.check_policy`.
    policy:
        Batas waktu dan izin format yang berlaku.
    feature_names:
        Dipakai kalau artifact tidak menyimpan ``feature_names_in_``.
    train_end:
        Akhir data latih menurut user; tidak bisa diverifikasi aplikasi.
    """

    def __init__(
        self,
        path: str | Path,
        policy: LoadPolicy,
        *,
        name: str = "uploaded_sklearn",
        label: str = "Model scikit-learn",
        feature_names: Sequence[str] | None = None,
        signal_kind: SignalKind | None = None,
        train_end: pd.Timestamp | str | None = None,
        threshold: float = 0.5,
        allow_short: bool = False,
        sizing: PositionSizing = "binary",
    ):
        self.path = Path(path)
        self.policy = policy
        self.name = name
        self.label = label
        self.threshold = threshold
        self.allow_short = allow_short
        self.sizing: PositionSizing = sizing
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None

        info = run_worker(self.path, "describe", policy)
        self._estimator = info["estimator"]
        self._expected = info["n_features"]
        self._params = info["params"] or {}

        stored = tuple(info["feature_names"]) if info["feature_names"] else None
        chosen = tuple(feature_names) if feature_names else None
        self._feature_names = stored or chosen
        self._names_are_declared_by_user = stored is None and chosen is not None

        if self._feature_names and self._expected is not None:
            if len(self._feature_names) != self._expected:
                raise ValueError(
                    f"artifact mengharapkan {self._expected} fitur, "
                    f"tapi {len(self._feature_names)} dipilih"
                )

        if signal_kind is not None:
            self.signal_kind: SignalKind = signal_kind
        elif info["has_predict_proba"]:
            self.signal_kind = "proba"
        elif info["is_classifier"]:
            self.signal_kind = "class"
        else:
            self.signal_kind = "expected_return"

    # -- kontrak ModelAdapter -------------------------------------------------

    def describe(self) -> ModelSpec:
        notes = []
        if self._names_are_declared_by_user:
            notes.append(
                "Nama fitur tidak tersimpan di artifact; urutan yang dipakai "
                "adalah pilihanmu dan tidak bisa diverifikasi aplikasi."
            )
        return ModelSpec(
            name=self.name,
            label=self.label,
            params=self.param_specs(),
            signal_kind=self.signal_kind,
            feature_names=self._feature_names,
            train_end=self.train_end,
            requires_fit=False,
            description=". ".join([f"{self._estimator} dari {self.path.name}", *notes]),
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError(
            "artifact yang diunggah dipakai apa adanya; melatih ulang berarti "
            "membuang bobot yang sudah ada di dalamnya"
        )

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        matrix, usable = self._align(X)
        values = np.full(len(X), np.nan, dtype=float)
        if usable.any():
            predicted = run_worker(
                self.path, "predict", self.policy,
                matrix=matrix, kind=self.signal_kind,
            )
            values[usable] = np.asarray(predicted, dtype=float)

        return to_weights(
            values,
            self.signal_kind,
            index=X.index,
            threshold=self.threshold,
            allow_short=self.allow_short,
            sizing=self.sizing,
        )

    # -- internal -------------------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        return self._feature_names

    def param_specs(self) -> dict[str, ParamSpec]:
        """Hyperparameter artifact, sekadar untuk ditampilkan.

        Semuanya hanya bisa dibaca: mengubahnya tidak akan mengubah apa pun,
        karena modelnya sudah terlatih dengan nilai-nilai itu. Panel merender
        keduanya berbeda supaya user tidak mengira slider di sini melatih ulang.
        """
        return infer_param_specs(self._params, skip=HIDDEN_PARAMS)

    def _align(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._feature_names:
            missing = [name for name in self._feature_names if name not in X.columns]
            if missing:
                raise ValueError(
                    f"artifact butuh fitur {missing} yang tidak ada; "
                    f"tersedia: {list(X.columns)}"
                )
            frame = X[list(self._feature_names)]
        else:
            frame = X

        if self._expected is not None and frame.shape[1] != self._expected:
            raise ValueError(
                f"artifact mengharapkan {self._expected} kolom, dapat {frame.shape[1]}"
            )

        matrix = frame.to_numpy(dtype=float)
        usable = ~np.isnan(matrix).any(axis=1)
        return matrix[usable], usable
