"""Katalog gabungan model yang bisa dipilih dari satu dropdown.

Tier 1 menghasilkan classifier tabular yang dilatih di sini; Tier 3 menghasilkan
peramal deret waktu yang dipakai zero-shot. Keduanya keluar sebagai
:class:`~stocklab.models.base.ModelAdapter` yang sama, jadi UI tidak perlu
membedakannya — yang perlu dibedakan cuma fitur yang mereka butuhkan, dan itu
sudah dinyatakan tiap model lewat ``describe().feature_names``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from stocklab.models.forecaster import FORECASTERS, build_forecaster
from stocklab.models.registry import MODELS, build_model
from stocklab.params import ParamSpec

__all__ = ["CatalogEntry", "build_any", "catalog"]


@dataclass(frozen=True)
class CatalogEntry:
    """Satu pilihan di dropdown model."""

    name: str
    label: str
    description: str
    kind: str
    """``"tabular"`` dilatih di sini lewat walk-forward; ``"forecast"`` zero-shot."""

    def param_specs(self) -> dict[str, ParamSpec]:
        if self.kind == "forecast":
            return FORECASTERS[self.name].param_specs()
        return MODELS[self.name].param_specs()


def catalog() -> dict[str, CatalogEntry]:
    """Seluruh model bawaan, tabular lebih dulu lalu peramal."""
    entries: dict[str, CatalogEntry] = {}
    for name, entry in MODELS.items():
        entries[name] = CatalogEntry(
            name=name, label=entry.label, description=entry.description,
            kind="tabular",
        )
    for name, entry in FORECASTERS.items():
        entries[name] = CatalogEntry(
            name=name, label=entry.label, description=entry.description,
            kind="forecast",
        )
    return entries


def build_any(name: str, params: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
    """Bangun adapter apa pun jenisnya, dari namanya saja."""
    if name in FORECASTERS:
        return build_forecaster(name, params, **kwargs)
    return build_model(name, params, **kwargs)
