"""Proses terpisah yang memuat artifact pickle dan memakainya sekali.

Dijalankan sebagai ``python -m stocklab.models.worker <request> <response>``.
Permintaan dan jawaban lewat berkas, bukan pipe, supaya tidak ada urusan
buffering dan supaya isinya bisa diperiksa saat ada yang salah.

Apa yang sebenarnya diberikan pemisahan ini
-------------------------------------------
Yang **didapat**: model yang rusak, memakan memori tanpa henti, atau
menggantung tidak menjatuhkan aplikasi; prosesnya bisa dibunuh saat lewat
waktu, dan sisa keadaan apa pun ikut mati bersamanya.

Yang **tidak didapat**: perlindungan terhadap berkas yang sengaja dibuat jahat.
``pickle`` menjalankan kode saat deserialisasi, dan kode itu berjalan dengan
hak akses user yang sama — berkas dan jaringan yang sama. Di Windows batas
memori juga tidak tersedia karena modul ``resource`` tidak ada.

Menyebut ini "sandbox" akan menyesatkan. Namanya proses pekerja, dan gunanya
membatasi kecelakaan.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _describe(model: Any) -> dict[str, Any]:
    """Kumpulkan yang bisa digali dari estimator tanpa menjalankannya."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        steps = getattr(model, "steps", None)
        if steps:
            names = getattr(steps[0][1], "feature_names_in_", None)

    params: dict[str, Any] = {}
    try:
        raw = model.get_params(deep=False)
    except (AttributeError, TypeError):
        raw = {}
    for key, value in raw.items():
        # Hanya nilai sederhana yang bisa menyeberang dengan aman dan berarti
        # di panel parameter; estimator bersarang dilewati.
        if isinstance(value, (bool, int, float, str, type(None))):
            params[key] = value

    n_features = getattr(model, "n_features_in_", None)
    return {
        "estimator": type(model).__name__,
        "feature_names": [str(n) for n in names] if names is not None else None,
        "n_features": int(n_features) if n_features is not None else None,
        "params": params,
        "has_predict_proba": hasattr(model, "predict_proba"),
        "is_classifier": getattr(model, "_estimator_type", None) == "classifier",
    }


def _predict(model: Any, matrix: Any, kind: str) -> Any:
    import numpy as np

    if kind == "proba" and hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(matrix), dtype=float)
        if proba.ndim == 1:
            return proba
        classes = getattr(model, "classes_", None)
        if classes is None:
            return proba[:, -1]
        classes = np.asarray(classes)
        column = int(np.flatnonzero(classes == classes.max())[0])
        return proba[:, column]
    return np.asarray(model.predict(matrix), dtype=float)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: worker <request.pkl> <response.pkl>", file=sys.stderr)
        return 2

    request_path, response_path = Path(argv[0]), Path(argv[1])
    response: dict[str, Any]
    try:
        with request_path.open("rb") as handle:
            request = pickle.load(handle)

        import joblib

        model = joblib.load(request["artifact"])
        command = request.get("command", "describe")

        if command == "describe":
            response = {"ok": True, "result": _describe(model)}
        elif command == "predict":
            values = _predict(model, request["matrix"], request.get("kind", "proba"))
            response = {"ok": True, "result": values}
        else:
            response = {"ok": False, "error": f"perintah tidak dikenal: {command!r}"}
    except BaseException as error:  # termasuk SystemExit dari artifact bermasalah
        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    try:
        with response_path.open("wb") as handle:
            pickle.dump(response, handle, protocol=4)
    except Exception as error:  # jawaban tidak bisa ditulis; induk akan melihatnya
        print(f"gagal menulis jawaban: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
