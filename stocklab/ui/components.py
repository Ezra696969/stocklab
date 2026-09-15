"""Potongan tampilan kecil yang dipakai berulang di beberapa panel."""

from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

import streamlit as st

from stocklab.ui.theme import PALETTE

__all__ = ["concerns", "glass", "metric_row", "panel_title"]


def glass(name: str):
    """Wadah bergaya kaca.

    Namanya menjadi kelas ``.st-key-glass-<nama>`` yang diincar CSS tema.
    """
    return st.container(key=f"glass-{name}")


def panel_title(text: str, note: str | None = None) -> None:
    st.markdown(f'<p class="panel-title">{html.escape(text)}</p>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="panel-note">{html.escape(note)}</p>', unsafe_allow_html=True)


def metric_row(items: Sequence[tuple[str, str, str | None]], accent: str | None = None) -> None:
    """Deretan kartu metrik.

    Tiap item berupa ``(label, nilai, keterangan)``. ``accent`` menyebut label
    mana yang diberi warna strategi, dipakai untuk angka utama panel.
    """
    cards = []
    for label, value, sub in items:
        classes = "metric is-accent" if label == accent else "metric"
        subtitle = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
        cards.append(
            f'<div class="{classes}">'
            f'<div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div>'
            f"{subtitle}</div>"
        )
    st.markdown(f'<div class="metric-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def concerns(items: Iterable[str], empty_message: str | None = None) -> None:
    """Daftar kekhawatiran validasi.

    Daftar kosong tidak ditampilkan sebagai kelulusan. Kalimat penggantinya
    menyebutkan dengan tepat apa yang sebenarnya diketahui: pemeriksaan yang
    dijalankan tidak menemukan masalah yang dicarinya.
    """
    found = list(items)
    if not found:
        if empty_message:
            st.markdown(
                f'<div class="clear">{html.escape(empty_message)}</div>',
                unsafe_allow_html=True,
            )
        return

    for concern in found:
        st.markdown(
            f'<div class="concern"><span class="mark">!</span>'
            f"<span>{html.escape(concern)}</span></div>",
            unsafe_allow_html=True,
        )


def fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if number != number else f"{number * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if number != number else f"{number:.{digits}f}"


def fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if number != number else f"{number:,.0f}"
