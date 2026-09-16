"""Adapter untuk model ONNX.

ONNX adalah graf operasi yang di-parse, bukan program yang dijalankan. Memuatnya
tidak mengeksekusi kode dari berkas, sehingga ini satu-satunya jalur upload yang
benar-benar aman — dan karena itu jadi default.

Harganya: nama fitur tidak selalu ikut tersimpan. Konverter seperti ``skl2onnx``
secara bawaan menghasilkan satu input tensor bernama ``float_input`` dengan
bentuk ``[None, n_fitur]``, sehingga yang diketahui hanya *berapa* fitur yang
dibutuhkan, bukan fitur apa saja dan dalam urutan apa.

Kalau begitu keadaannya, adapter ini melaporkan jumlahnya dan berhenti di situ.
Menebak fitur mana yang dimaksud, atau menganggap urutan yang dipilih user
kebetulan benar, adalah cara paling halus menghasilkan backtest yang salah
tanpa satu pun error muncul. User yang memilih urutannya sendiri, dan aplikasi
menyatakan dengan jelas bahwa urutan itu tidak bisa diverifikasi.

Nama fitur bisa disertakan saat ekspor lewat ``metadata_props``; kalau ada,
adapter memakainya dan urutannya jadi pasti.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stocklab.models.base import ModelSpec, PositionSizing, SignalKind, to_weights

__all__ = ["OnnxAdapter", "read_onnx_metadata"]

#: Kunci metadata tempat nama fitur dicari, sesuai urutan prioritas.
FEATURE_NAME_KEYS = ("feature_names", "feature_names_in_", "features", "columns")


def read_onnx_metadata(path: str | Path) -> dict[str, Any]:
    """Baca signature input dan metadata sebuah berkas ONNX.

    Hanya membaca; tidak menjalankan grafnya.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # jangan cetak peringatan ke konsol aplikasi
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("model ONNX ini tidak punya input")
    if len(inputs) > 1:
        raise ValueError(
            f"model ONNX ini punya {len(inputs)} input terpisah; yang didukung "
            "baru satu tensor fitur"
        )

    tensor = inputs[0]
    shape = list(tensor.shape)
    n_features = None
    if len(shape) == 2 and isinstance(shape[1], int):
        n_features = int(shape[1])

    meta = session.get_modelmeta()
    custom = dict(meta.custom_metadata_map or {})

    feature_names = None
    for key in FEATURE_NAME_KEYS:
        raw = custom.get(key)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = [part.strip() for part in str(raw).split(",") if part.strip()]
        if isinstance(parsed, (list, tuple)) and parsed:
            feature_names = tuple(str(name) for name in parsed)
            break

    return {
        "input_name": tensor.name,
        "input_type": tensor.type,
        "n_features": n_features,
        "feature_names": feature_names,
        "outputs": tuple(output.name for output in session.get_outputs()),
        "producer": meta.producer_name or None,
        "metadata": custom,
        "session": session,
    }


class OnnxAdapter:
    """Bungkus model ONNX agar memenuhi :class:`ModelAdapter`.

    Parameters
    ----------
    path:
        Berkas ONNX yang sudah lolos :func:`~stocklab.models.artifact.check_policy`.
    feature_names:
        Dipakai kalau berkasnya tidak menyimpan nama fitur. Jumlahnya harus
        sama dengan yang diharapkan model.
    train_end:
        Akhir data latih menurut user. Aplikasi tidak punya cara memverifikasi
        ini; lihat :attr:`~stocklab.models.base.ModelSpec.train_end`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "uploaded_onnx",
        label: str = "Model ONNX",
        feature_names: Sequence[str] | None = None,
        signal_kind: SignalKind | None = None,
        train_end: pd.Timestamp | str | None = None,
        threshold: float = 0.5,
        allow_short: bool = False,
        sizing: PositionSizing = "binary",
    ):
        self.path = Path(path)
        self.name = name
        self.label = label
        self.threshold = threshold
        self.allow_short = allow_short
        self.sizing: PositionSizing = sizing
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None

        info = read_onnx_metadata(self.path)
        self._session = info["session"]
        self._input_name = info["input_name"]
        self._input_type = info["input_type"]
        self._expected = info["n_features"]
        self._producer = info["producer"]
        self._outputs = info["outputs"]

        stored = info["feature_names"]
        chosen = tuple(feature_names) if feature_names else None
        self._feature_names = stored or chosen
        self._names_are_declared_by_user = stored is None and chosen is not None

        if self._feature_names and self._expected is not None:
            if len(self._feature_names) != self._expected:
                raise ValueError(
                    f"model mengharapkan {self._expected} fitur, "
                    f"tapi {len(self._feature_names)} dipilih"
                )

        self.signal_kind: SignalKind = signal_kind or self._infer_signal_kind()

    # -- kontrak ModelAdapter -------------------------------------------------

    def describe(self) -> ModelSpec:
        notes = []
        if self._names_are_declared_by_user:
            notes.append(
                "Nama fitur tidak tersimpan di berkas; urutan yang dipakai adalah "
                "pilihanmu dan tidak bisa diverifikasi aplikasi."
            )
        return ModelSpec(
            name=self.name,
            label=self.label,
            params={},  # graf ONNX tidak menyimpan hyperparameter yang bisa diatur
            signal_kind=self.signal_kind,
            feature_names=self._feature_names,
            train_end=self.train_end,
            requires_fit=False,
            description=self._description(notes),
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError(
            "model ONNX adalah graf terlatih dan dipakai apa adanya; "
            "melatih ulang berarti membuang bobot yang diunggah"
        )

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        matrix = self._align(X)
        raw = self._run(matrix)
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
    def n_features(self) -> int | None:
        return self._expected

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        return self._feature_names

    def _description(self, notes: list[str]) -> str:
        parts = [f"Model ONNX dari {self.path.name}"]
        if self._producer:
            parts.append(f"dihasilkan {self._producer}")
        if self._expected is not None:
            parts.append(f"{self._expected} fitur")
        return ". ".join([", ".join(parts), *notes])

    def _infer_signal_kind(self) -> SignalKind:
        """Tebak bentuk keluaran dari nama output grafnya.

        Konverter scikit-learn menamai output kedua ``output_probability`` untuk
        classifier. Kalau tidak ada penanda apa pun, keluarannya diperlakukan
        sebagai expected return -- pilihan yang paling tidak merusak, karena
        yang dipakai hanya tandanya.
        """
        joined = " ".join(self._outputs).lower()
        if "probab" in joined:
            return "proba"
        if "label" in joined or "class" in joined:
            return "class"
        return "expected_return"

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        """Susun matriks fitur sesuai yang diharapkan model."""
        if self._feature_names:
            missing = [name for name in self._feature_names if name not in X.columns]
            if missing:
                raise ValueError(
                    f"model butuh fitur {missing} yang tidak ada; "
                    f"tersedia: {list(X.columns)}"
                )
            frame = X[list(self._feature_names)]
        else:
            frame = X

        if self._expected is not None and frame.shape[1] != self._expected:
            raise ValueError(
                f"model mengharapkan {self._expected} kolom, dapat {frame.shape[1]}"
            )

        dtype = np.float64 if "double" in self._input_type else np.float32
        return np.ascontiguousarray(frame.to_numpy(dtype=dtype))

    def _run(self, matrix: np.ndarray) -> np.ndarray:
        """Jalankan graf dan ambil kolom yang berarti.

        Baris dengan NaN dilewati, bukan dikirim ke model: ONNX Runtime akan
        menerimanya dan mengembalikan angka yang terlihat wajar padahal tak
        bermakna. Barisnya dikembalikan sebagai NaN, yang oleh engine
        diperlakukan sebagai posisi kas.
        """
        usable = ~np.isnan(matrix).any(axis=1)
        output = np.full(len(matrix), np.nan, dtype=float)
        if not usable.any():
            return output

        results = self._session.run(None, {self._input_name: matrix[usable]})
        output[usable] = self._extract(results)
        return output

    def _extract(self, results: list[Any]) -> np.ndarray:
        """Ambil deret angka dari keluaran graf.

        Classifier scikit-learn yang dikonversi menghasilkan label di keluaran
        pertama dan probabilitas di keluaran kedua, kadang sebagai daftar dict
        per baris alih-alih matriks.
        """
        if self.signal_kind == "proba":
            for candidate in results[1:] + results[:1]:
                values = self._as_probability(candidate)
                if values is not None:
                    return values
        flat = np.asarray(results[0], dtype=float).reshape(len(results[0]), -1)
        return flat[:, -1] if flat.shape[1] > 1 else flat[:, 0]

    @staticmethod
    def _as_probability(candidate: Any) -> np.ndarray | None:
        """Ubah keluaran probabilitas jadi peluang kelas positif."""
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            keys = sorted(candidate[0])
            positive = keys[-1]
            return np.array([float(row[positive]) for row in candidate])

        array = np.asarray(candidate)
        if array.ndim == 2 and array.shape[1] >= 2:
            return array[:, -1].astype(float)
        return None
