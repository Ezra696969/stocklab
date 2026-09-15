"""Menyusun konteks untuk chatbot, dan narasi cadangan tanpa LLM.

Aturan yang mengikat seluruh modul ini: **model bahasa tidak pernah menghitung.**
Python yang menghitung seluruh metrik, hasilnya dikirim ke prompt sebagai data,
dan tugas model hanya menjelaskannya. Llama 3.2 yang diminta menaksir Sharpe dari
grafik akan menjawab dengan angka yang terdengar meyakinkan dan salah, dan user
tidak punya cara membedakannya dari yang benar.

:func:`fallback_narrative` menulis penjelasan yang sama tanpa LLM sama sekali.
Bukan sekadar penanganan error: peninjau yang membuka proyek ini kemungkinan
besar tidak memasang Ollama, dan panel chat yang mati total di mesin mereka
sama saja dengan tidak ada.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["SYSTEM_PROMPT", "build_context", "fallback_narrative", "format_context"]


SYSTEM_PROMPT = """Kamu asisten analisis backtesting di aplikasi StockLab.

Aturan yang tidak boleh dilanggar:
- JANGAN PERNAH mengarang atau menaksir angka. Semua angka sudah dihitung dan
  diberikan kepadamu sebagai data. Kalau sebuah angka tidak ada di data, katakan
  angka itu belum dihitung.
- Jangan memberi saran investasi, rekomendasi beli/jual, atau proyeksi harga.
  Kamu menjelaskan hasil eksperimen, bukan menasihati keputusan keuangan.
- Bersikap skeptis terhadap hasil yang bagus. Kalau ada kekhawatiran di data,
  sebutkan, bahkan ketika user terdengar senang dengan hasilnya.
- Jawab dalam Bahasa Indonesia, ringkas, 2-4 kalimat kecuali diminta lebih rinci.
- Jelaskan istilah statistik dengan bahasa biasa saat pertama kali muncul.
"""


def _clean(value: Any) -> Any:
    """Ubah nilai numpy dan NaN menjadi sesuatu yang aman untuk JSON."""
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if np.isnan(number) else round(number, 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def build_context(
    *,
    metrics: Mapping[str, Any] | None = None,
    model_name: str | None = None,
    model_params: Mapping[str, Any] | None = None,
    features: Sequence[str] | None = None,
    walkforward: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    trials: Mapping[str, Any] | None = None,
    period: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Kumpulkan segala yang sudah dihitung menjadi satu struktur data.

    Hanya bagian yang benar-benar ada yang disertakan, sehingga model bisa
    membedakan "nilainya nol" dari "belum dijalankan".
    """
    context: dict[str, Any] = {}
    if model_name:
        context["model"] = {"name": model_name, "params": _clean(model_params or {})}
    if features:
        context["features"] = list(features)
    if period:
        context["period"] = _clean(period)
    if metrics:
        context["metrics"] = _clean(metrics)
    if walkforward:
        context["walkforward"] = _clean(walkforward)
    if validation:
        context["validation"] = _clean(validation)
    if trials:
        context["search"] = _clean(trials)
    return context


def format_context(context: Mapping[str, Any]) -> str:
    """Ubah konteks menjadi teks yang ringkas untuk prompt.

    JSON mentah boros token dan membuat model kecil mudah tersesat; daftar
    berlabel lebih mudah diikutinya.
    """
    if not context:
        return "Belum ada backtest yang dijalankan."

    lines: list[str] = []

    model = context.get("model")
    if model:
        params = ", ".join(f"{k}={v}" for k, v in (model.get("params") or {}).items())
        lines.append(f"Model: {model['name']}" + (f" ({params})" if params else ""))

    if context.get("features"):
        lines.append(f"Fitur: {', '.join(context['features'])}")

    period = context.get("period")
    if period:
        lines.append(
            f"Periode: {period.get('start')} sampai {period.get('end')} "
            f"({period.get('n_bars')} bar), biaya {period.get('cost_bps')} bps"
        )

    metrics = context.get("metrics") or {}
    if metrics:
        lines.append("\nHasil (semua sudah dihitung, jangan diubah):")
        for key, label in _METRIC_LABELS.items():
            if key in metrics and metrics[key] is not None:
                lines.append(f"  {label}: {_format_metric(key, metrics[key])}")

    wf = context.get("walkforward")
    if wf:
        lines.append(
            f"\nWalk-forward: dilatih ulang {wf.get('n_fitted')} kali, "
            f"prediksi pertama {wf.get('first_prediction')}, "
            f"cakupan {_pct(wf.get('coverage'))}"
        )

    validation = context.get("validation") or {}
    if validation:
        lines.append("\nValidasi:")
        rotation = validation.get("rotation")
        if rotation:
            lines.append(
                f"  Uji rotasi: p = {rotation.get('p_value')}, "
                f"sharpe asli {rotation.get('observed')} vs "
                f"maksimum acak {rotation.get('null_max')}"
            )
        costs = validation.get("costs")
        if costs and costs.get("breakeven_bps") is not None:
            lines.append(f"  Titik impas biaya: {costs['breakeven_bps']:.1f} bps")
        for name in ("parameters", "periods"):
            block = validation.get(name)
            if block:
                lines.append(
                    f"  Stabilitas {name}: {int(block.get('n_runs', 0))} run, "
                    f"unggul nominal {_pct(block.get('win_rate'))}, "
                    f"unggul sharpe {_pct(block.get('sharpe_win_rate'))}"
                )
        if validation.get("deflated_sharpe") is not None:
            lines.append(f"  Deflated Sharpe: {validation['deflated_sharpe']}")

    search = context.get("search")
    if search:
        lines.append(
            f"\nUser sudah mencoba {int(search.get('n_distinct', 0))} konfigurasi "
            f"berbeda dalam sesi ini."
        )

    concerns = (validation or {}).get("concerns") or []
    if concerns:
        lines.append("\nKekhawatiran yang sudah terdeteksi:")
        lines.extend(f"  - {concern}" for concern in concerns)

    return "\n".join(lines)


_METRIC_LABELS = {
    "total_return": "Return total",
    "cagr": "CAGR",
    "benchmark_cagr": "CAGR pasif",
    "sharpe": "Sharpe",
    "benchmark_sharpe": "Sharpe pasif",
    "sortino": "Sortino",
    "max_drawdown": "Drawdown terdalam",
    "benchmark_max_drawdown": "Drawdown pasif",
    "calmar": "Calmar",
    "hit_rate": "Hit rate",
    "exposure": "Eksposur",
    "turnover_per_year": "Turnover per tahun",
    "total_cost": "Total biaya",
    "alpha": "Alpha",
    "beta": "Beta",
    "information_ratio": "Information ratio",
}

_AS_PERCENT = {
    "total_return",
    "cagr",
    "benchmark_cagr",
    "max_drawdown",
    "benchmark_max_drawdown",
    "hit_rate",
    "exposure",
    "alpha",
}


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.0f}%"


def _format_metric(key: str, value: Any) -> str:
    number = float(value)
    if key in _AS_PERCENT:
        return f"{number * 100:.2f}%"
    if key in ("total_cost",):
        return f"{number:,.0f}"
    return f"{number:.3f}"


def fallback_narrative(context: Mapping[str, Any]) -> str:
    """Tulis ringkasan tanpa LLM sama sekali.

    Dipakai kalau Ollama tidak terpasang. Nadanya lebih datar daripada jawaban
    model, tapi isinya sama dan angkanya berasal dari sumber yang sama persis.
    """
    metrics = context.get("metrics") or {}
    if not metrics:
        return (
            "Belum ada backtest yang dijalankan. Pilih model dan indikator di panel "
            "bawah, lalu tekan Jalankan."
        )

    sharpe = metrics.get("sharpe")
    benchmark = metrics.get("benchmark_sharpe")
    ratio = metrics.get("equity_ratio")
    parts: list[str] = []

    if sharpe is not None and benchmark is not None:
        verdict = "di atas" if sharpe > benchmark else "di bawah"
        parts.append(
            f"Sharpe strategi {sharpe:.2f}, {verdict} pasif yang {benchmark:.2f}."
        )
    if ratio is not None:
        outcome = "mengungguli" if ratio > 1 else "tertinggal dari"
        parts.append(
            f"Secara nominal equity akhirnya {outcome} buy & hold "
            f"({ratio:.2f}x)."
        )
    if metrics.get("exposure") is not None and metrics.get("turnover_per_year") is not None:
        parts.append(
            f"Posisi terbuka {float(metrics['exposure']) * 100:.0f}% waktu dengan "
            f"turnover {float(metrics['turnover_per_year']):.0f} kali setahun."
        )
    if metrics.get("max_drawdown") is not None:
        parts.append(
            f"Penurunan terdalam dari puncak {float(metrics['max_drawdown']) * 100:.1f}%."
        )

    concerns = (context.get("validation") or {}).get("concerns") or []
    if concerns:
        parts.append("\n\nYang perlu diperhatikan:")
        parts.extend(f"\n- {concern}" for concern in concerns)
    elif context.get("validation"):
        parts.append(
            "\n\nPemeriksaan validasi tidak menemukan masalah yang dicarinya. Itu "
            "bukan berarti strategi ini bagus, hanya berarti masalah yang diperiksa "
            "tidak ditemukan."
        )

    search = context.get("search")
    if search and int(search.get("n_distinct", 0)) > 5:
        parts.append(
            f"\n\nCatatan: sudah {int(search['n_distinct'])} konfigurasi berbeda "
            f"dicoba di sesi ini. Semakin banyak dicoba, semakin besar peluang "
            f"salah satunya terlihat bagus karena kebetulan."
        )

    return " ".join(parts).strip()
