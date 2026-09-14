import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ollama
import yfinance as yf

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(page_title="AI Trading Portofolio", layout="wide", page_icon="📈")

# ============================================================
# 2. CUSTOM CSS — TEMA DASHBOARD GELAP (mengikuti referensi UI)
# ============================================================
CUSTOM_CSS = """
<style>
:root{
    --bg: #101114;
    --panel: #1a1c22;
    --panel-2: #1e2128;
    --border: rgba(255,255,255,0.06);
    --text: #f2f2f3;
    --muted: #8b8d98;
    --orange: #f5943d;
    --orange-soft: rgba(245,148,61,0.15);
    --green: #3ecf6e;
    --green-soft: rgba(62,207,110,0.15);
    --red: #ef5b5b;
    --red-soft: rgba(239,91,91,0.15);
}

html, body, [data-testid="stAppViewContainer"]{
    background: radial-gradient(circle at 20% 0%, #17181d 0%, var(--bg) 55%) !important;
    color: var(--text);
}
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stSidebar"]{ background: var(--panel); border-right: 1px solid var(--border); }
.block-container{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

/* ---- Top nav bar ---- */
.topnav{
    display:flex; align-items:center; justify-content:space-between;
    background: var(--panel); border:1px solid var(--border); border-radius: 18px;
    padding: 14px 22px; margin-bottom: 22px;
}
.topnav .brand{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.05rem; }
.topnav .brand-badge{
    width:34px; height:34px; border-radius:10px; background: var(--orange);
    display:flex; align-items:center; justify-content:center; font-size:18px;
}
.topnav .menu{ display:flex; gap:6px; }
.topnav .menu span{
    padding:8px 16px; border-radius:10px; font-size:0.86rem; color: var(--muted);
}
.topnav .menu span.active{ background: var(--orange-soft); color: var(--orange); font-weight:600; }
.topnav .user{ display:flex; align-items:center; gap:10px; }
.topnav .avatar{
    width:36px; height:36px; border-radius:50%; background: linear-gradient(135deg,var(--orange),#ffb26b);
    display:flex; align-items:center; justify-content:center; font-weight:700; color:#1a1a1a;
}
.topnav .user-meta{ font-size:0.78rem; color: var(--muted); line-height:1.1; }
.topnav .user-meta b{ color: var(--text); font-size:0.86rem; }

/* ---- Section title ---- */
.section-title{ display:flex; align-items:center; justify-content:space-between; margin: 4px 0 16px 0; }
.section-title h2{ font-size:1.4rem; margin:0; }
.pill-tabs span{
    padding:8px 18px; border-radius:10px; font-size:0.82rem; margin-left:8px; color: var(--muted);
    border:1px solid var(--border);
}
.pill-tabs span.active{ background: var(--orange); color:#161616; font-weight:700; border:none; }

/* ---- Metric cards ---- */
.metric-card{
    background: var(--panel); border:1px solid var(--border); border-radius: 18px;
    padding: 18px 20px; height: 100%;
}
.metric-card .label{ color: var(--muted); font-size:0.82rem; margin-bottom:10px; }
.metric-card .value-row{ display:flex; align-items:baseline; gap:10px; }
.metric-card .value{ font-size:1.65rem; font-weight:700; color: var(--text); }
.badge{
    display:inline-flex; align-items:center; gap:4px; font-size:0.74rem; font-weight:700;
    padding: 3px 9px; border-radius: 999px;
}
.badge-green{ background: var(--green-soft); color: var(--green); }
.badge-red{ background: var(--red-soft); color: var(--red); }

/* ---- Generic panel card ---- */
.panel-card{
    background: var(--panel); border:1px solid var(--border); border-radius: 20px;
    padding: 20px 22px; margin-bottom: 20px;
}
.panel-card h4{ margin:0 0 14px 0; font-size:1.02rem; display:flex; align-items:center; justify-content:space-between; }
.panel-card h4 .dots{ color: var(--muted); }

/* ---- Chat panel ---- */
.chat-card{ background: var(--panel); border:1px solid var(--border); border-radius: 20px; padding: 18px 18px 8px 18px; }
.chat-card h4{ margin:0 0 4px 0; font-size:1.05rem; }
.chat-card .sub{ color: var(--muted); font-size:0.82rem; margin-bottom: 12px; }

/* ---- Streamlit widget restyle ---- */
div[data-baseweb="select"] > div{
    background: var(--panel-2) !important; border-color: var(--border) !important; border-radius: 12px !important;
}
.stSlider [data-baseweb="slider"]{ padding-top: 6px; }
.stButton > button{
    background: var(--orange) !important; color:#171717 !important; border:none !important;
    border-radius: 12px !important; font-weight:700 !important; padding: 10px 0 !important;
}
.stButton > button:hover{ background:#ffab5e !important; }
[data-testid="stFileUploader"], .stMultiSelect{
    background: transparent;
}
hr{ border-color: var(--border) !important; }
h1, h2, h3, h4, h5{ color: var(--text) !important; }
[data-testid="stMetricValue"]{ color: var(--text); }
.stChatMessage{ background: var(--panel-2) !important; border-radius: 14px !important; border:1px solid var(--border) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge(value):
    """Membuat badge hijau/merah berdasarkan tanda nilai."""
    cls = "badge-green" if value >= 0 else "badge-red"
    arrow = "↑" if value >= 0 else "↓"
    return f'<span class="badge {cls}">{value:+.2f}% {arrow}</span>'


def metric_card(label, value_str, delta_val=None):
    delta_html = badge(delta_val) if delta_val is not None else ""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value-row">
            <div class="value">{value_str}</div>
            {delta_html}
        </div>
    </div>
    """


# ============================================================
# 3. TOP NAVBAR
# ============================================================
st.markdown("""
<div class="topnav">
    <div class="brand">
        <div class="brand-badge">📈</div>
        AI Trading Portofolio
    </div>
    <div class="menu">
        <span class="active">Dashboard</span>
        <span>Backtest</span>
        <span>Riwayat</span>
        <span>Pengaturan</span>
    </div>
    <div class="user">
        <div class="user-meta"><b>Ezra</b><br/>Quant Trader</div>
        <div class="avatar">E</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_kiri, col_kanan = st.columns([7, 3])

# ==========================================
# PANEL KIRI: VISUALISASI & KONTROL MODEL
# ==========================================
with col_kiri:
    st.markdown("""
    <div class="section-title">
        <h2>Simulasi Model Machine Learning</h2>
        <div class="pill-tabs"><span class="active">Cabinet Saya</span><span>Portofolio</span><span>Statistik</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel-card"><h4>Pengaturan Model <span class="dots">⚙️</span></h4>', unsafe_allow_html=True)
    col_model, col_tuning, col_fitur = st.columns(3)

    with col_model:
        ticker_saham = st.selectbox(
            "Pilih Saham:",
            ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "AAPL", "NVDA"]
        )
        pilihan_model = st.selectbox("Pilih Algoritma:", ["Random Forest", "Logistic Regression"])
        tombol_jalankan = st.button("Jalankan Model 🚀", use_container_width=True)

    with col_tuning:
        window_rsi = st.slider("Periode RSI (Hari)", 7, 30, 14)
        window_sma = st.slider("Periode SMA Pendek", 10, 50, 20)

    with col_fitur:
        st.file_uploader("Import Data Fitur Tambahan (.csv)", type=['csv'])
        st.multiselect("Gunakan Fitur:", ["RSI", "MACD", "Volume Change"], default=["RSI", "MACD"])
    st.markdown('</div>', unsafe_allow_html=True)

    # === LOGIKA SAAT TOMBOL DITEKAN ===
    if tombol_jalankan:
        with st.spinner(f'Mengunduh data {ticker_saham} dan melatih model {pilihan_model}...'):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            df = yf.download(ticker_saham, start="2020-01-01")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['Return_1d'] = df['Close'].pct_change()

            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            fitur = ['Return_1d', 'MACD', 'RSI']
            X = df[fitur]
            y = df['Target']

            batas_waktu = '2025-01-01'
            X_train = X[X.index < batas_waktu]
            X_test = X[X.index >= batas_waktu]
            y_train = y[y.index < batas_waktu]
            y_test = y[y.index >= batas_waktu]

            if pilihan_model == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
            elif pilihan_model == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            else:
                model = RandomForestClassifier(random_state=42)

            model.fit(X_train, y_train)
            prediksi = model.predict(X_test)

            df_test = df[df.index >= batas_waktu].copy()
            df_test['Prediksi'] = prediksi
            df_test['Strategy_Return'] = df_test['Return_1d'] * df_test['Prediksi'].shift(1)
            df_test.dropna(inplace=True)

            modal_awal = 10000000
            df_test['Total_Uang_Pasif'] = modal_awal * (1 + df_test['Return_1d']).cumprod()
            df_test['Total_Uang_Strategi'] = modal_awal * (1 + df_test['Strategy_Return']).cumprod()

            # Grafik utama — palet oranye/hijau sesuai referensi
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_test.index, y=df_test['Total_Uang_Pasif'],
                mode='lines', name='Beli & Diam (Pasif)',
                line=dict(color='#8b8d98', dash='dash', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_test.index, y=df_test['Total_Uang_Strategi'],
                mode='lines', name=f'Strategi ML: {pilihan_model}',
                line=dict(color='#f5943d', width=3)
            ))
            fig.update_layout(
                title=f"Kinerja Model {pilihan_model} ({ticker_saham}) vs Beli & Diam",
                xaxis_title="Tanggal", yaxis_title="Total Nilai Uang (Rp)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified", legend=dict(orientation="h", y=1.1),
                margin=dict(l=10, r=10, t=60, b=10),
            )

            return_pasif = (df_test['Total_Uang_Pasif'].iloc[-1] - modal_awal) / modal_awal * 100
            return_strategi = (df_test['Total_Uang_Strategi'].iloc[-1] - modal_awal) / modal_awal * 100
            rata_untung = df_test['Strategy_Return'].mean()
            volatilitas = df_test['Strategy_Return'].std()
            sharpe = (rata_untung / volatilitas * np.sqrt(252)) if volatilitas != 0 else 0

            # Donut fitur (importance untuk RF, |koefisien| untuk LR) — meniru panel "Active listings"
            try:
                if pilihan_model == "Random Forest":
                    importances = model.feature_importances_
                else:
                    importances = np.abs(model.coef_[0])
                importances = importances / importances.sum() * 100
            except Exception:
                importances = [34, 33, 33]

            donut = go.Figure(data=[go.Pie(
                labels=fitur, values=importances, hole=0.62,
                marker=dict(colors=['#f5943d', '#3ecf6e', '#c9cbd4']),
                textinfo='none',
            )])
            donut.update_layout(
                showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0), height=190,
            )

            st.session_state['metrik_terakhir'] = {
                'model': pilihan_model, 'ticker': ticker_saham,
                'return_pasif': return_pasif, 'return_strategi': return_strategi,
                'sharpe': sharpe, 'modal_akhir': df_test['Total_Uang_Strategi'].iloc[-1],
            }
            st.session_state['grafik_terakhir'] = fig
            st.session_state['donut_terakhir'] = donut
            st.session_state['fitur_terakhir'] = list(zip(fitur, importances))

    # === METRIC CARDS (mengikuti kartu "Properties visits / Deals completed / ...") ===
    if 'metrik_terakhir' in st.session_state:
        m = st.session_state['metrik_terakhir']
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Return Pasif", f"{m['return_pasif']:.1f}%", m['return_pasif']), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Return Strategi", f"{m['return_strategi']:.1f}%", m['return_strategi']), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Sharpe Ratio", f"{m['sharpe']:.2f}", m['sharpe']), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Modal Akhir", f"Rp {m['modal_akhir']:,.0f}", m['return_strategi']), unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # === GRAFIK UTAMA + DONUT (mengikuti "Mortgage payment" / "Active listings") ===
    chart_col, donut_col = st.columns([7, 3])
    with chart_col:
        st.markdown('<div class="panel-card"><h4>Kinerja Strategi <span class="dots">···</span></h4>', unsafe_allow_html=True)
        if 'grafik_terakhir' in st.session_state:
            st.plotly_chart(st.session_state['grafik_terakhir'], use_container_width=True)
        else:
            st.info("Tekan tombol 'Jalankan Model' untuk melihat simulasi.")
        st.markdown('</div>', unsafe_allow_html=True)

    with donut_col:
        st.markdown('<div class="panel-card"><h4>Kontribusi Fitur</h4>', unsafe_allow_html=True)
        if 'donut_terakhir' in st.session_state:
            st.plotly_chart(st.session_state['donut_terakhir'], use_container_width=True)
            colors = ['#f5943d', '#3ecf6e', '#c9cbd4']
            for (nama, pct), warna in zip(st.session_state['fitur_terakhir'], colors):
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:6px;'>"
                    f"<span>● {nama}</span><span style='color:{warna};font-weight:700'>{pct:.1f}%</span></div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Jalankan model untuk melihat kontribusi fitur.")
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PANEL KANAN: AI CHATBOT PEMANTAU
# ==========================================
with col_kanan:
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown("<h4>🤖 AI Assistant</h4>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Asisten ini memantau performa modelmu.</div>", unsafe_allow_html=True)

    wadah_chat = st.container(height=500)

    if "pesan_chat" not in st.session_state:
        st.session_state.pesan_chat = [
            {"role": "assistant", "content": "Halo! Jalankan simulasi di sebelah kiri, lalu tanyakan analisisnya kepada saya."}
        ]

    with wadah_chat:
        for pesan in st.session_state.pesan_chat:
            with st.chat_message(pesan["role"]):
                st.write(pesan["content"])

    prompt_user = st.chat_input("Tanya AI tentang performa model...")
    if prompt_user:
        st.session_state.pesan_chat.append({"role": "user", "content": prompt_user})
        with wadah_chat:
            with st.chat_message("user"):
                st.write(prompt_user)

            konteks_sistem = "Kamu adalah asisten Quantitative Analyst profesional yang ahli di bidang Data Science saham. Jawablah dalam bahasa Indonesia dengan jelas, padat, dan analitis. "

            if 'metrik_terakhir' in st.session_state:
                data = st.session_state['metrik_terakhir']
                konteks_sistem += (
                    f"Saat ini, pengguna sedang menguji algoritma {data['model']} pada saham {data['ticker']}. "
                    f"Hasil simulasinya: Total Untung Strategi adalah {data['return_strategi']:.2f}%, "
                    f"Sharpe Ratio adalah {data['sharpe']:.2f}. Sebagai perbandingan, strategi pasif "
                    f"(Beli & Diam) menghasilkan untung {data['return_pasif']:.2f}%. "
                    "Gunakan data ini untuk menjawab pertanyaan pengguna jika relevan."
                )
            else:
                konteks_sistem += "Pengguna belum menjalankan model apa pun. Sarankan mereka untuk menekan tombol 'Jalankan Model' di panel kiri terlebih dahulu."

            pesan_untuk_ai = [{"role": "system", "content": konteks_sistem}]
            pesan_untuk_ai.extend(st.session_state.pesan_chat)

            with st.chat_message("assistant"):
                wadah_balasan = st.empty()
                wadah_balasan.write("Sedang menganalisis data...")
                try:
                    respons = ollama.chat(model='llama3.2', messages=pesan_untuk_ai)
                    balasan_ai = respons['message']['content']
                    wadah_balasan.write(balasan_ai)
                    st.session_state.pesan_chat.append({"role": "assistant", "content": balasan_ai})
                except Exception:
                    wadah_balasan.error("Gagal menghubungi AI lokal. Pastikan aplikasi Ollama sedang berjalan di latar belakang.")
    st.markdown('</div>', unsafe_allow_html=True)
