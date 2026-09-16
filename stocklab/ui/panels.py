"""Kelima panel aplikasi, ditambah panel validasi.

Tiap fungsi di sini merender satu panel dan mengembalikan pilihan user. Tidak
ada yang menghitung apa pun sendiri — perhitungan hidup di
:mod:`stocklab.ui.pipeline` dan di paket intinya.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocklab.features import FEATURES, FeatureConfig
from stocklab.models import catalog, get_model_entry
from stocklab.params import ParamSpec
from stocklab.ui.components import (
    concerns,
    fmt_money,
    fmt_num,
    fmt_pct,
    glass,
    metric_row,
    panel_title,
)
from stocklab.ui.pipeline import RunResult
from stocklab.ui.theme import PALETTE, plotly_layout

__all__ = [
    "chart_panel",
    "cross_section_panel",
    "chat_panel",
    "indicator_panel",
    "model_panel",
    "param_panel",
    "validation_panel",
]


# ---------------------------------------------------------------------------
# Kiri atas: equity curve
# ---------------------------------------------------------------------------


def chart_panel(result: RunResult | None) -> None:
    """Equity curve strategi melawan pasif."""
    with glass("chart"):
        if result is None:
            panel_title("Kurva Ekuitas")
            st.markdown(
                '<div style="height:360px;display:flex;align-items:center;'
                'justify-content:center;color:rgba(255,255,255,0.38);'
                'font-size:0.88rem;text-align:center;line-height:1.7">'
                "Pilih model dan indikator di panel bawah,<br>lalu tekan Jalankan."
                "</div>",
                unsafe_allow_html=True,
            )
            return

        period = result.period()
        summary = result.walkforward_summary()
        note = (
            f"{period['start']} sampai {period['end']} · {period['n_bars']} bar · "
            f"biaya {period['cost_bps']:.0f} bps · dilatih ulang "
            f"{summary['n_fitted']}x"
        )
        if result.skipped:
            note += (
                f" · {result.skipped} bar awal tidak dihitung karena model "
                f"belum boleh berpendapat"
            )
        panel_title("Kurva Ekuitas", note)
        st.plotly_chart(
            _equity_figure(result), width="stretch", config={"displayModeBar": False}
        )
        _headline_metrics(result)


def _equity_figure(result: RunResult) -> go.Figure:
    equity = result.backtest.equity
    passive = result.backtest.benchmark_equity

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=passive.index,
            y=passive.to_numpy(),
            name="Pasif",
            mode="lines",
            line={"color": PALETTE["passive"], "width": 2, "dash": "dot"},
            hovertemplate="Pasif %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.to_numpy(),
            name="Strategi",
            mode="lines",
            line={"color": PALETTE["strategy"], "width": 2},
            hovertemplate="Strategi %{y:,.0f}<extra></extra>",
        )
    )

    layout = plotly_layout(height=372)
    # Sumbu dikunci ke rentang data, bukan ke nol. Equity tidak pernah mendekati
    # nol, jadi sumbu yang dipaksa mulai dari sana akan memampatkan seluruh
    # selisih antara kedua garis menjadi pita tipis di bagian atas.
    low = min(float(equity.min()), float(passive.min()))
    high = max(float(equity.max()), float(passive.max()))
    padding = (high - low) * 0.08 or high * 0.02
    layout["yaxis"] = {**layout["yaxis"], "range": [low - padding, high + padding]}
    # Label langsung di ujung garis: identitas tidak boleh bergantung warna saja.
    layout["annotations"] = [
        _end_label(equity, "Strategi", PALETTE["strategy"]),
        _end_label(passive, "Pasif", PALETTE["passive"]),
    ]
    figure.update_layout(**layout)
    figure.update_yaxes(tickformat=",.0f")
    return figure


def _end_label(series: pd.Series, text: str, color: str) -> dict[str, Any]:
    return {
        "x": series.index[-1],
        "y": float(series.iloc[-1]),
        "text": f"  {text}",
        "showarrow": False,
        "xanchor": "left",
        "font": {"color": color, "size": 11},
    }


def _headline_metrics(result: RunResult) -> None:
    metrics = result.metrics
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    metric_row(
        [
            (
                "Ekuitas Akhir",
                fmt_money(result.backtest.equity.iloc[-1]),
                f"pasif {fmt_money(result.backtest.benchmark_equity.iloc[-1])}",
            ),
            ("Sharpe", fmt_num(metrics["sharpe"]), f"pasif {fmt_num(metrics['benchmark_sharpe'])}"),
            (
                "Drawdown",
                fmt_pct(metrics["max_drawdown"]),
                f"pasif {fmt_pct(metrics['benchmark_max_drawdown'])}",
            ),
            ("Eksposur", fmt_pct(metrics["exposure"], 0), f"turnover {fmt_num(metrics['turnover_per_year'], 0)}x/th"),
            ("Biaya", fmt_money(metrics["total_cost"]), f"beta {fmt_num(metrics['beta'])}"),
        ],
        accent="Ekuitas Akhir",
    )


# ---------------------------------------------------------------------------
# Kanan atas: chatbot
# ---------------------------------------------------------------------------


def chat_panel(client, context: Mapping[str, Any], messages: list[dict[str, str]]) -> str | None:
    """Panel chat. Mengembalikan pertanyaan baru kalau user mengirim satu."""
    with glass("chat"):
        panel_title(
            "Asisten",
            client.status.explain() if not client.available else None,
        )

        history = st.container(height=396)
        with history:
            if not messages:
                st.chat_message("assistant").markdown(client.greeting(context))
            for message in messages:
                st.chat_message(message["role"]).markdown(message["content"])

        return st.chat_input("Tanya tentang hasil backtest...")


# ---------------------------------------------------------------------------
# Kiri bawah: pilihan model
# ---------------------------------------------------------------------------


def model_panel() -> tuple[str, dict[str, Any]]:
    """Pilihan model dan pengaturan sizing."""
    with glass("model"):
        panel_title("Model")

        entries = catalog()
        names = list(entries)
        chosen = st.selectbox(
            "Algoritma",
            names,
            index=names.index("random_forest") if "random_forest" in names else 0,
            format_func=lambda name: entries[name].label,
            key="model_name",
        )
        st.caption(entries[chosen].description)
        if entries[chosen].kind == "forecast":
            st.info(
                "Peramal deret waktu membaca riwayat harga, bukan indikator. "
                "Pilih **Harga** di panel Indikator; indikator lain diabaikan."
            )

        sizing = st.radio(
            "Ukuran posisi",
            ["binary", "scaled"],
            horizontal=True,
            key="sizing",
            format_func=lambda mode: "Penuh / kas" if mode == "binary" else "Ikut keyakinan",
            help=(
                "Penuh/kas: masuk sepenuhnya atau tidak sama sekali. "
                "Ikut keyakinan: prediksi yang ragu mendapat modal lebih kecil."
            ),
        )
        threshold = st.slider(
            "Ambang keyakinan", 0.50, 0.80, 0.50, 0.01,
            key="threshold",
            help="Menaikkannya membuat model lebih pemilih dan menurunkan turnover.",
        )

        upload = _upload_section()

        return chosen, {"sizing": sizing, "threshold": threshold, "upload": upload}


def _upload_section() -> dict[str, Any]:
    """Bagian upload artifact (Tier 2).

    Izin pickle disajikan sebagai pilihan sadar, bukan kotak centang yang
    tersembunyi di balik kata "lanjutan". User yang menyalakannya harus membaca
    apa konsekuensinya lebih dulu.
    """
    from stocklab.models import LoadPolicy

    with st.expander("Model sendiri"):
        allow_pickle = st.checkbox(
            "Izinkan format pickle (.joblib / .pkl)",
            value=False,
            key="allow_pickle",
        )
        policy = LoadPolicy(allow_pickle=allow_pickle)
        if allow_pickle:
            st.warning(policy.describe_risk())
        else:
            st.caption(policy.describe_risk())

        uploaded = st.file_uploader(
            "Artifact model",
            type=["onnx", "joblib", "pkl"],
            key="model_upload",
        )

        declared_end = st.date_input(
            "Data latih berakhir (opsional)",
            value=None,
            key="train_end",
            help=(
                "Kalau diisi, backtest dimulai setelah tanggal ini. Aplikasi "
                "tidak punya cara memeriksa apakah model sudah pernah melihat "
                "periode yang diuji; kalau dikosongkan, hasilnya jadi "
                "tanggung jawabmu."
            ),
        )

    return {"file": uploaded, "policy": policy, "train_end": declared_end}


# ---------------------------------------------------------------------------
# Tengah bawah: tuning indikator
# ---------------------------------------------------------------------------


def indicator_panel() -> tuple[FeatureConfig, ...]:
    """Pemilihan dan penyetelan indikator.

    Panel ini tidak pernah menyebut RSI atau SMA. Isinya dibangun dari registry,
    jadi indikator yang ditambahkan ke registry langsung muncul di sini.
    """
    with glass("indicators"):
        panel_title("Indikator")

        default = ["rsi", "roc", "macd"]
        chosen = st.multiselect(
            "Dipakai sebagai fitur",
            sorted(FEATURES),
            default=[name for name in default if name in FEATURES],
            format_func=lambda name: FEATURES[name].label,
            key="feature_names",
        )

        configs: list[FeatureConfig] = []
        for name in chosen:
            spec = FEATURES[name]
            with st.expander(spec.label, expanded=len(chosen) <= 2):
                st.caption(spec.description)
                configs.append(
                    FeatureConfig(name, _render_params(spec.params, prefix=f"feat_{name}"))
                )

        if not configs:
            st.caption("Pilih minimal satu indikator.")
        return tuple(configs)


# ---------------------------------------------------------------------------
# Kanan bawah: parameter model
# ---------------------------------------------------------------------------


def param_panel(model_name: str) -> dict[str, Any]:
    """Parameter model, dirender dari spec-nya sendiri.

    Tidak ada satu pun nama parameter yang ditulis di sini. Yang dibaca adalah
    :class:`~stocklab.params.ParamSpec` milik model, sehingga jalur yang sama
    nanti melayani model yang di-upload user.
    """
    with glass("params"):
        entry = catalog()[model_name]
        panel_title("Parameter Model", entry.label)

        specs = entry.param_specs()
        values = _render_params(specs, prefix=f"param_{model_name}")

        hidden = [name for name, spec in specs.items() if not spec.is_editable]
        if hidden:
            st.caption(
                f"{len(hidden)} parameter tidak ditampilkan karena tipenya tidak "
                f"bisa diwakili satu kontrol: {', '.join(hidden)}."
            )
        return values


def _render_params(specs: Mapping[str, ParamSpec], prefix: str) -> dict[str, Any]:
    """Ubah spec parameter menjadi kontrol Streamlit.

    Slider hanya dipakai kalau kedua batasnya diketahui. Parameter tanpa batas
    mendapat kotak angka: slider dengan rentang karangan terlihat berwibawa
    padahal isinya asal, dan user tidak punya cara membedakannya.
    """
    values: dict[str, Any] = {}
    for name, spec in specs.items():
        key = f"{prefix}_{name}"
        label = spec.display_label

        if spec.kind == "bool":
            values[name] = st.checkbox(label, value=bool(spec.default), key=key, help=spec.help)
        elif spec.kind == "choice" and spec.choices:
            options = list(spec.choices)
            index = options.index(spec.default) if spec.default in options else 0
            values[name] = st.selectbox(
                label, options, index=index, key=key, help=spec.help,
                format_func=lambda value: "otomatis" if value is None else str(value),
            )
        elif spec.kind == "int" and spec.is_bounded:
            values[name] = st.slider(
                label, int(spec.low), int(spec.high), int(spec.default),
                int(spec.step or 1), key=key, help=spec.help,
            )
        elif spec.kind == "float" and spec.is_bounded:
            values[name] = st.slider(
                label, float(spec.low), float(spec.high), float(spec.default),
                float(spec.step or 0.01), key=key, help=spec.help,
            )
        elif spec.kind in ("int", "float"):
            values[name] = st.number_input(
                label, value=spec.default, key=key, help=spec.help
            )
        # kind == "opaque" sengaja dilewati: tidak ada kontrol yang jujur untuknya.
    return values


# ---------------------------------------------------------------------------
# Validasi
# ---------------------------------------------------------------------------


def validation_panel(report, trial_count: int) -> None:
    """Hasil pemeriksaan yang berusaha mementahkan backtest."""
    with glass("validation"):
        panel_title(
            "Validasi",
            f"{trial_count} konfigurasi berbeda dicoba di sesi ini"
            if trial_count > 1
            else None,
        )

        if report is None:
            st.caption(
                "Belum dijalankan. Pemeriksaan ini melatih ulang model puluhan kali, "
                "jadi butuh waktu lebih lama daripada satu backtest biasa."
            )
            return

        concerns(
            report.concerns,
            empty_message=(
                "Pemeriksaan yang dijalankan tidak menemukan masalah yang dicarinya. "
                "Itu bukan berarti strategi ini bagus."
            ),
        )

        tabs = st.tabs(["Rotasi", "Biaya", "Parameter", "Periode"])

        with tabs[0]:
            rotation = report.rotation
            if rotation is None:
                st.caption("Tidak dijalankan.")
            else:
                st.markdown(
                    "Sinyal digeser waktunya ratusan kali. Eksposur dan turnover "
                    "tetap sama, tapi keselarasan dengan harga hancur — jadi yang "
                    "tersisa cuma bentuk posisi tanpa timing."
                )
                metric_row(
                    [
                        ("Sharpe Asli", fmt_num(rotation.observed), None),
                        ("Acak Tertinggi", fmt_num(rotation.null_max), f"dari {rotation.n_rotations} geseran"),
                        ("p-value", f"{rotation.p_value:.4f}", "makin kecil makin meyakinkan"),
                    ],
                    accent="Sharpe Asli",
                )

        with tabs[1]:
            costs = report.costs
            if costs is None:
                st.caption("Tidak dijalankan.")
            else:
                breakeven = costs.breakeven_bps
                st.markdown(
                    f"Titik impas **{breakeven:.1f} bps** per sisi transaksi."
                    if breakeven is not None
                    else "Strategi tidak pernah mengungguli pasif pada tingkat biaya mana pun."
                )
                st.dataframe(
                    costs.table[["sharpe", "equity_ratio", "total_cost"]].rename(
                        columns={
                            "sharpe": "Sharpe",
                            "equity_ratio": "Rasio vs Pasif",
                            "total_cost": "Total Biaya",
                        }
                    ),
                    width="stretch",
                )

        with tabs[2]:
            _stability_tab(
                report.parameters,
                "Model dilatih ulang untuk tiap kombinasi parameter. Hasil yang "
                "runtuh begitu satu slider digeser adalah tanda model menghafal.",
            )

        with tabs[3]:
            _stability_tab(
                report.periods,
                "Alur yang sama dijalankan pada beberapa rentang waktu. Ini yang "
                "paling sering mementahkan strategi yang lolos pemeriksaan lain.",
                skipped_reason=(
                    "Rentang tanggalnya terlalu pendek untuk dibagi menjadi "
                    "beberapa jendela yang bisa dibandingkan. Perpanjang rentang "
                    "di panel kiri — minimal tiga tahun — lalu jalankan ulang."
                ),
            )

        if report.deflated_sharpe == report.deflated_sharpe:
            st.markdown(
                f'<div style="margin-top:14px;font-size:0.82rem;color:rgba(255,255,255,0.58)">'
                f"Deflated Sharpe <b style='color:rgba(255,255,255,0.92)'>"
                f"{report.deflated_sharpe:.3f}</b> — peluang Sharpe ini benar-benar "
                f"positif setelah memperhitungkan {report.n_trials_counted} konfigurasi "
                f"yang dicoba.</div>",
                unsafe_allow_html=True,
            )


def _stability_tab(report, explanation: str, skipped_reason: str | None = None) -> None:
    """Render satu tab stabilitas.

    Pemeriksaan yang dilewati harus menyebutkan alasannya. Keterangan "tidak
    dijalankan" tanpa sebab membuat user mengira pemeriksaannya sudah lewat,
    padahal justru tidak pernah berjalan.
    """
    if report is None:
        st.caption(skipped_reason or "Tidak dijalankan pada run ini.")
        return

    st.markdown(explanation)
    spread = report.sharpe_spread()
    metric_row(
        [
            ("Run", str(report.n_runs), f"{report.n_failed} gagal" if report.n_failed else None),
            ("Sharpe Terburuk", fmt_num(spread["min"]), f"median {fmt_num(spread['median'])}"),
            ("Unggul Nominal", fmt_pct(report.win_rate, 0), None),
            ("Unggul Sharpe", fmt_pct(report.sharpe_win_rate, 0), None),
        ],
        accent="Sharpe Terburuk",
    )
    st.dataframe(report.table, width="stretch", height=210)


# ---------------------------------------------------------------------------
# Lintas saham
# ---------------------------------------------------------------------------


def cross_section_panel(report) -> None:
    """Hasil alur yang sama dijalankan pada beberapa aset.

    Panel ini sengaja tidak menyorot aset terbaik. Melaporkan pemenang dari
    sepuluh yang dicoba adalah multiple testing yang paling mudah dilakukan
    tanpa sadar; yang ditampilkan seluruh sebarannya, diurutkan supaya yang
    terburuk sama terlihatnya dengan yang terbaik.
    """
    with glass("universe"):
        panel_title("Lintas Saham")

        if report is None:
            st.caption(
                "Belum dijalankan. Pemeriksaan ini mengulang seluruh alur untuk "
                "tiap aset, jadi paling lama di antara semuanya — tapi juga yang "
                "paling sering membantah hasil yang lolos pemeriksaan lain."
            )
            return

        concerns(
            report.concerns(),
            empty_message=(
                "Metode ini bertahan di sebagian besar aset yang diuji. Itu bukan "
                "jaminan, hanya berarti hasilnya tidak bergantung pada satu saham."
            ),
        )

        summary = report.summary()
        metric_row(
            [
                (
                    "Unggul Sharpe",
                    fmt_pct(summary["sharpe_win_rate"], 0),
                    f"nominal {fmt_pct(summary['win_rate'], 0)}",
                ),
                (
                    "Sharpe Median",
                    fmt_num(summary["median_sharpe"]),
                    f"terburuk {fmt_num(summary['worst_sharpe'])}",
                ),
                (
                    "Aset",
                    f"{summary['n_assets']:.0f}",
                    f"{summary['n_failed']:.0f} gagal" if summary["n_failed"] else None,
                ),
                (
                    "Pengamatan Bebas",
                    fmt_num(summary["effective_sample_size"], 1),
                    f"korelasi {fmt_num(summary['mean_correlation'])}",
                ),
            ],
            accent="Unggul Sharpe",
        )

        st.plotly_chart(
            _cross_section_figure(report),
            width="stretch",
            config={"displayModeBar": False},
        )

        with st.expander("Angka per aset"):
            st.dataframe(report.table, width="stretch")


def _cross_section_figure(report) -> go.Figure:
    """Sharpe tiap aset, dengan pasif sebagai penanda pembanding."""
    table = report.table.dropna(subset=["sharpe"]).sort_values("sharpe")

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=table["sharpe"],
            y=table.index,
            orientation="h",
            name="Strategi",
            marker={"color": PALETTE["strategy"]},
            hovertemplate="%{y} · strategi %{x:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["benchmark_sharpe"],
            y=table.index,
            mode="markers",
            name="Pasif",
            marker={"color": PALETTE["passive"], "size": 9,
                    "line": {"color": "#0B0B0F", "width": 1.5}},
            hovertemplate="%{y} · pasif %{x:.2f}<extra></extra>",
        )
    )

    layout = plotly_layout(height=max(240, 34 * len(table)))
    layout["margin"] = {"l": 8, "r": 20, "t": 10, "b": 8}
    layout["xaxis"] = {**layout["xaxis"], "title": {"text": "Sharpe"},
                       "zeroline": True, "zerolinecolor": PALETTE["grid"]}
    layout["yaxis"] = {**layout["yaxis"], "gridcolor": "rgba(0,0,0,0)"}
    figure.update_layout(**layout)
    return figure
