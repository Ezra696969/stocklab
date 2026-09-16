"""Pintu masuk tunggal untuk artifact yang diunggah user.

Satu fungsi, :func:`load_uploaded_model`, yang memutuskan format, menegakkan
kebijakan, lalu memilih adapter. Panel UI memanggil ini dan tidak perlu tahu
bedanya ONNX dengan pickle; begitu juga sisa aplikasi, karena keduanya keluar
sebagai :class:`~stocklab.models.base.ModelAdapter` yang sama.

Urutannya penting: kebijakan diperiksa sebelum pustaka pemuat mana pun menyentuh
berkas. Berkas yang ditolak tidak pernah dibuka.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from stocklab.models.artifact import (
    ArtifactInfo,
    ArtifactRejected,
    LoadPolicy,
    check_policy,
)
from stocklab.models.base import PositionSizing, SignalKind

__all__ = ["inspect_uploaded", "load_uploaded_model"]


def load_uploaded_model(
    path: str | Path,
    *,
    policy: LoadPolicy | None = None,
    feature_names: Sequence[str] | None = None,
    signal_kind: SignalKind | None = None,
    train_end: pd.Timestamp | str | None = None,
    threshold: float = 0.5,
    allow_short: bool = False,
    sizing: PositionSizing = "binary",
) -> Any:
    """Muat artifact dan kembalikan adapter yang siap dipakai.

    Raises
    ------
    ArtifactRejected
        Berkasnya melanggar kebijakan; berkas tidak pernah dibuka.
    SandboxError
        Artifact pickle gagal dimuat, menggantung, atau melempar error.
    """
    policy = policy or LoadPolicy()
    path = Path(path)
    fmt = check_policy(path, policy)

    shared = {
        "feature_names": feature_names,
        "signal_kind": signal_kind,
        "train_end": train_end,
        "threshold": threshold,
        "allow_short": allow_short,
        "sizing": sizing,
    }

    if fmt == "onnx":
        from stocklab.models.onnx_adapter import OnnxAdapter

        return OnnxAdapter(path, **shared)

    from stocklab.models.sandbox import SandboxedAdapter

    return SandboxedAdapter(path, policy, **shared)


def inspect_uploaded(
    path: str | Path, policy: LoadPolicy | None = None
) -> ArtifactInfo:
    """Gali isi artifact tanpa memakainya untuk memprediksi.

    Dipakai panel upload untuk menampilkan fitur apa yang dibutuhkan model dan
    berapa banyak, sebelum user menekan Jalankan. Untuk artifact pickle,
    penggalian ini pun terjadi di proses terpisah.
    """
    policy = policy or LoadPolicy()
    path = Path(path)
    fmt = check_policy(path, policy)
    size = path.stat().st_size

    if fmt == "onnx":
        from stocklab.models.onnx_adapter import read_onnx_metadata

        meta = read_onnx_metadata(path)
        notes = []
        if meta["feature_names"] is None:
            notes.append(
                "Berkas ini tidak menyimpan nama fitur, jadi kamu yang memilih "
                "fitur beserta urutannya. Urutan yang keliru tidak akan memicu "
                "error, hanya hasil yang salah."
            )
        return ArtifactInfo(
            path=path,
            format=fmt,
            size_bytes=size,
            feature_names=meta["feature_names"],
            n_features=meta["n_features"],
            estimator=meta["producer"],
            signal_kind="proba" if "probab" in " ".join(meta["outputs"]).lower()
            else "expected_return",
            notes=tuple(notes),
        )

    from stocklab.models.sandbox import run_worker

    found = run_worker(path, "describe", policy)
    notes = []
    if not found["feature_names"]:
        notes.append(
            "Artifact ini dilatih tanpa nama kolom, jadi kamu yang memilih fitur "
            "beserta urutannya. Urutan yang keliru tidak akan memicu error, "
            "hanya hasil yang salah."
        )
    notes.append(
        "Berkas format pickle menjalankan kodenya sendiri saat dimuat. "
        "Pemuatannya dipisahkan ke proses lain untuk membatasi kerusakan akibat "
        "kecelakaan, bukan sebagai pengaman terhadap berkas yang dibuat jahat."
    )
    return ArtifactInfo(
        path=path,
        format=fmt,
        size_bytes=size,
        feature_names=tuple(found["feature_names"]) if found["feature_names"] else None,
        n_features=found["n_features"],
        params=found["params"],
        estimator=found["estimator"],
        signal_kind="proba" if found["has_predict_proba"]
        else ("class" if found["is_classifier"] else "expected_return"),
        notes=tuple(notes),
    )
