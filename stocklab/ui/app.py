"""Aplikasi StockLab.

Berkas ini menata panel dan meneruskan pilihan user ke
:mod:`stocklab.ui.pipeline`. Tidak ada perhitungan di sini — kalau ada baris di
modul ini yang menghitung metrik, baris itu salah tempat.

Tata letaknya mengikuti PLAN.md bagian 2: kurva ekuitas di kiri atas, asisten di
kanan atas, lalu tiga panel kontrol berjajar di bawahnya.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from stocklab.data import DEFAULT_END, DEFAULT_START, load_csv, load_ohlcv
from stocklab.llm.client import ChatClient
from stocklab.llm.context import build_context
from stocklab.ui import panels
from stocklab.ui.components import glass, panel_title
from stocklab.ui.pipeline import RunConfig, make_experiment, run_pipeline
from stocklab.ui.state import Trial, session_defaults
from stocklab.ui.theme import CSS
from stocklab.validation import validate

PRESET_TICKERS = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "^JKSE", "AAPL", "MSFT"]


def main() -> None:
    st.set_page_config(
        page_title="StockLab", page_icon="◐", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    for key, value in session_defaults().items():
        st.session_state.setdefault(key, value)

    settings = _sidebar()
    st.markdown("# StockLab")
    st.markdown(
        '<p style="color:rgba(255,255,255,0.52);margin-top:-12px;font-size:0.9rem">'
        "Lab backtesting model ML — walk-forward, biaya transaksi, dan pemeriksaan "
        "yang berusaha mementahkan hasilnya.</p>",
        unsafe_allow_html=True,
    )

    result = st.session_state["result"]
    client = _chat_client()
    context = _context(result)

    top_left, top_right = st.columns([7, 3], gap="medium")
    with top_left:
        panels.chart_panel(result)
    with top_right:
        question = panels.chat_panel(client, context, st.session_state["messages"])

    bottom_left, bottom_mid, bottom_right = st.columns(3, gap="medium")
    with bottom_left:
        model_name, sizing = panels.model_panel()
    with bottom_mid:
        features = panels.indicator_panel()
    with bottom_right:
        model_params = panels.param_panel(model_name)

    config = RunConfig(
        features=features,
        model_name=model_name,
        model_params=model_params,
        threshold=sizing["threshold"],
        sizing=sizing["sizing"],
        **settings["run"],
    )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    run_col, validate_col, _ = st.columns([1, 1, 4])
    run_clicked = run_col.button("Jalankan", type="primary", width="stretch")
    validate_clicked = validate_col.button(
        "Validasi", width="stretch", disabled=result is None
    )

    panels.validation_panel(
        st.session_state["validation"], st.session_state["trials"].n_distinct
    )

    if run_clicked:
        _run(settings, config)
    if validate_clicked:
        _validate(settings, config)
    if question:
        _answer(question, client, context)


def _sidebar() -> dict:
    with st.sidebar:
        st.markdown("### Data")
        source = st.radio("Sumber", ["Ticker", "Berkas CSV"], horizontal=True)
        ticker = csv_file = None
        if source == "Ticker":
            ticker = st.selectbox("Simbol", PRESET_TICKERS)
        else:
            csv_file = st.file_uploader("CSV harga", type=["csv"])

        start, end = st.columns(2)
        start_date = start.date_input("Mulai", date.fromisoformat(DEFAULT_START))
        end_date = end.date_input("Sampai", date.fromisoformat(DEFAULT_END))

        st.markdown("### Backtest")
        capital = st.number_input("Modal awal", 1_000, 10_000_000_000, 100_000, 10_000)
        cost = st.slider("Biaya transaksi (bps)", 0.0, 60.0, 10.0, 0.5)
        slippage = st.slider("Slippage (bps)", 0.0, 30.0, 5.0, 0.5)
        st.caption(
            "Keduanya per satu sisi transaksi. Strategi ber-turnover tinggi sangat "
            "sensitif terhadap angka ini — jalankan Validasi untuk melihat titik impasnya."
        )

        st.markdown("### Walk-forward")
        train = st.slider("Jendela latih (bar)", 60, 756, 252, 21)
        test = st.slider("Latih ulang tiap (bar)", 5, 126, 21, 1)
        expanding = st.toggle("Jendela membesar", value=True)
        horizon = st.slider("Horizon prediksi (bar)", 1, 10, 1)
        st.caption(
            f"Embargo mengikuti horizon ({horizon} bar) supaya label tidak "
            "membocorkan periode uji ke data latih."
        )

        trials = st.session_state["trials"]
        if len(trials):
            st.markdown("---")
            st.caption(
                f"{len(trials)} run, {trials.n_distinct} konfigurasi berbeda. "
                "Jumlah ini masuk ke koreksi deflated Sharpe."
            )
            if st.button("Reset hitungan"):
                trials.clear()
                st.rerun()

    return {
        "source": source,
        "ticker": ticker,
        "csv": csv_file,
        "start": start_date,
        "end": end_date,
        "run": {
            "initial_capital": float(capital),
            "cost_bps": float(cost),
            "slippage_bps": float(slippage),
            "train_size": int(train),
            "test_size": int(test),
            "expanding": bool(expanding),
            "horizon": int(horizon),
        },
    }


@st.cache_data(show_spinner=False)
def _load_ticker(ticker: str, start: str, end: str):
    return load_ohlcv(ticker, start, end)


def _load(settings) -> "pd.DataFrame":
    start, end = str(settings["start"]), str(settings["end"])
    if settings["source"] == "Berkas CSV":
        if settings["csv"] is None:
            raise ValueError("unggah berkas CSV dulu, atau pilih sumber Ticker")
        return load_csv(settings["csv"]).loc[start:end]
    return _load_ticker(settings["ticker"], start, end)


def _run(settings, config: RunConfig) -> None:
    try:
        with st.spinner("Memuat data..."):
            ohlcv = _load(settings)
        with st.spinner("Melatih walk-forward..."):
            result = run_pipeline(ohlcv, config)
    except Exception as error:
        st.error(f"{type(error).__name__}: {error}")
        return

    st.session_state["result"] = result
    # Validasi lama menjelaskan run lama; menyisakannya akan menyesatkan.
    st.session_state["validation"] = None
    st.session_state["trials"].record(
        Trial(
            model=config.model_name,
            params=dict(config.model_params),
            features=result.feature_names,
            sharpe=result.metrics["sharpe"],
            equity_ratio=result.equity_ratio,
        )
    )
    st.rerun()


def _validate(settings, config: RunConfig) -> None:
    entry_grid = _grid_for(config)
    try:
        with st.spinner("Menjalankan pemeriksaan — model dilatih ulang puluhan kali..."):
            ohlcv = _load(settings)
            report = validate(
                make_experiment(ohlcv, config),
                grid=entry_grid,
                n_rotations=300,
                window_years=_window_years(ohlcv),
            )
    except Exception as error:
        st.error(f"{type(error).__name__}: {error}")
        return

    st.session_state["validation"] = report
    st.rerun()


def _window_years(ohlcv) -> int:
    """Panjang jendela stabilitas periode, menyesuaikan data yang tersedia.

    Jendela tetap tiga tahun membuat pemeriksaan ini diam-diam terlewat pada
    rentang default aplikasi, yang kurang beberapa hari dari tiga tahun. Yang
    dilewati justru pemeriksaan yang paling sering mementahkan strategi, dan
    user tidak diberi tahu apa pun — panelnya hanya menulis "tidak dijalankan".

    Dipilih jendela terpanjang yang masih menyisakan minimal dua jendela untuk
    dibandingkan; satu jendela tidak mengatakan apa-apa soal stabilitas.
    """
    span_years = (ohlcv.index[-1] - ohlcv.index[0]).days / 365.25
    for candidate in (3, 2):
        if span_years >= candidate + 1:
            return candidate
    return 2


def _grid_for(config: RunConfig) -> dict[str, list]:
    """Grid sempit di sekitar pengaturan user.

    Dibatasi tiga nilai per parameter numerik dan paling banyak dua parameter:
    tiap kombinasi berarti satu walk-forward penuh, dan grid yang murah hati di
    sini berarti menunggu bermenit-menit.
    """
    from stocklab.models import get_model_entry

    specs = get_model_entry(config.model_name).param_specs()
    grid: dict[str, list] = {}
    for name, spec in specs.items():
        if len(grid) == 2:
            break
        if spec.kind != "int" or not spec.is_bounded:
            continue
        current = int(config.model_params.get(name, spec.default))
        candidates = sorted(
            {
                max(int(spec.low), current // 2),
                current,
                min(int(spec.high), current * 2),
            }
        )
        if len(candidates) > 1:
            grid[name] = candidates
    return grid


def _chat_client() -> ChatClient:
    if "chat_client" not in st.session_state:
        st.session_state["chat_client"] = ChatClient()
    return st.session_state["chat_client"]


def _context(result) -> dict:
    if result is None:
        return build_context(trials=st.session_state["trials"].summary())

    report = st.session_state["validation"]
    return build_context(
        metrics={**result.metrics, "equity_ratio": result.equity_ratio},
        model_name=result.spec.label,
        model_params=result.spec.defaults(),
        features=result.feature_names,
        walkforward=result.walkforward_summary(),
        validation=report.summary() if report is not None else None,
        trials=st.session_state["trials"].summary(),
        period=result.period(),
    )


def _answer(question: str, client: ChatClient, context: dict) -> None:
    messages = st.session_state["messages"]
    messages.append({"role": "user", "content": question})
    with st.spinner("Menyusun jawaban..."):
        reply = client.reply(question, context, history=messages[:-1][-6:])
    messages.append({"role": "assistant", "content": reply})
    st.rerun()


if __name__ == "__main__":
    main()
