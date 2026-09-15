"""Deskripsi parameter yang bisa dirender jadi kontrol UI.

Satu tipe dipakai bersama oleh dua panel: tuning indikator (tengah-bawah) dan
parameter model (kanan-bawah). Keduanya menghadapi masalah yang sama —
mengambil sekumpulan parameter beserta tipe dan batasnya, lalu menampilkannya
sebagai kontrol yang benar — jadi tidak ada alasan menulisnya dua kali.

Catatan desain: rentang tidak pernah ditebak. Kalau batas sebuah parameter
tidak diketahui, ``low`` dan ``high`` dibiarkan ``None`` dan UI merender kotak
angka biasa, bukan slider. Slider dengan rentang karangan terlihat berwibawa
padahal isinya asal, dan user tidak punya cara membedakannya.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

__all__ = ["ParamSpec", "infer_param_specs", "specs_from_callable"]

ParamKind = Literal["int", "float", "bool", "choice", "opaque"]


@dataclass(frozen=True)
class ParamSpec:
    """Satu parameter yang bisa diatur, lengkap dengan cara merendernya."""

    name: str
    kind: ParamKind
    default: Any
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None
    label: str | None = None
    help: str | None = None

    @property
    def display_label(self) -> str:
        """Label untuk UI; jatuh ke nama parameter yang dirapikan."""
        return self.label or self.name.replace("_", " ").strip().capitalize()

    @property
    def is_editable(self) -> bool:
        """``False`` untuk parameter yang tidak bisa diwakili sebuah kontrol."""
        return self.kind != "opaque"

    @property
    def is_bounded(self) -> bool:
        """``True`` kalau kedua batas diketahui, sehingga slider layak dipakai."""
        return self.low is not None and self.high is not None

    def coerce(self, value: Any) -> Any:
        """Paksa ``value`` ke tipe parameter ini, atau lempar error."""
        if self.kind == "int":
            coerced = int(value)
        elif self.kind == "float":
            coerced = float(value)
        elif self.kind == "bool":
            coerced = bool(value)
        elif self.kind == "choice":
            coerced = value
        else:
            return value

        if self.kind == "choice" and self.choices is not None:
            if coerced not in self.choices:
                raise ValueError(
                    f"{self.name}={coerced!r} bukan pilihan yang sah; "
                    f"pilihan: {list(self.choices)}"
                )
        if self.kind in ("int", "float"):
            if self.low is not None and coerced < self.low:
                raise ValueError(f"{self.name}={coerced} di bawah batas {self.low}")
            if self.high is not None and coerced > self.high:
                raise ValueError(f"{self.name}={coerced} di atas batas {self.high}")
        return coerced

    def clamp(self, value: Any) -> Any:
        """Seperti :meth:`coerce`, tapi batas yang dilanggar ditarik ke tepi.

        Dipakai saat memuat konfigurasi lama yang batasnya sudah berubah, bukan
        untuk memvalidasi masukan baru.
        """
        if self.kind not in ("int", "float"):
            return self.coerce(value)
        number = float(value)
        if self.low is not None:
            number = max(number, float(self.low))
        if self.high is not None:
            number = min(number, float(self.high))
        return int(round(number)) if self.kind == "int" else number

    def with_bounds(
        self, low: float | None, high: float | None, step: float | None = None
    ) -> ParamSpec:
        """Salinan dengan batas baru; dipakai tabel hint per model."""
        return replace(self, low=low, high=high, step=step or self.step)


def _kind_of(value: Any) -> ParamKind:
    # bool dicek lebih dulu: bool adalah subclass int di Python.
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float" if math.isfinite(value) else "opaque"
    if isinstance(value, str):
        return "choice"
    return "opaque"


def infer_param_specs(
    params: Mapping[str, Any],
    *,
    overrides: Mapping[str, ParamSpec] | None = None,
    skip: Iterable[str] = (),
    choices: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, ParamSpec]:
    """Bangun spec dari pasangan nama/nilai default.

    Inilah jalur introspeksi yang mengisi panel parameter secara otomatis:
    ``get_params()`` milik scikit-learn menghasilkan persis bentuk mapping ini.

    Parameters
    ----------
    params:
        Nama parameter dan nilai defaultnya.
    overrides:
        Spec yang sudah disiapkan tangan, menang atas hasil tebakan. Di sinilah
        rentang yang benar-benar diketahui dimasukkan.
    skip:
        Nama yang tidak perlu ditampilkan (``random_state``, ``n_jobs``, dan
        sejenisnya).
    choices:
        Pilihan yang sah untuk parameter bertipe string.
    """
    overrides = dict(overrides or {})
    choices = dict(choices or {})
    skipped = set(skip)

    specs: dict[str, ParamSpec] = {}
    for name, default in params.items():
        if name in skipped:
            continue
        if name in overrides:
            specs[name] = overrides[name]
            continue

        kind = _kind_of(default)
        options = choices.get(name)
        if options is not None:
            specs[name] = ParamSpec(
                name=name, kind="choice", default=default, choices=tuple(options)
            )
            continue
        if kind == "choice" and options is None:
            # String tanpa daftar pilihan tidak bisa dirender dengan aman:
            # menebak opsinya sama saja mengarang API orang lain.
            specs[name] = ParamSpec(name=name, kind="opaque", default=default)
            continue

        specs[name] = ParamSpec(name=name, kind=kind, default=default)
    return specs


def specs_from_callable(
    fn: Callable[..., Any],
    *,
    overrides: Mapping[str, ParamSpec] | None = None,
    skip: Iterable[str] = (),
    choices: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, ParamSpec]:
    """Introspeksi parameter keyword sebuah fungsi beserta defaultnya.

    Parameter tanpa nilai default dilewati: itu data masukan (deret harga),
    bukan sesuatu yang diatur user.
    """
    signature = inspect.signature(fn)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return infer_param_specs(
        defaults, overrides=overrides, skip=skip, choices=choices
    )
