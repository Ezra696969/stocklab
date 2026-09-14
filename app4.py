import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ollama
import yfinance as yf
import base64

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(page_title="AI Trading Portofolio", layout="wide", page_icon="📈")

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return "" # Fallback jika gambar tidak ditemukan

# Pastikan gambar background Anda ada di folder yang sama
img_base64 = get_base64_of_bin_file('e705e6357257e7e7cf1466cec10a7389.jpg')

# ============================================================
# 2. INJEKSI SVG LIQUID GLASS FILTER
# ============================================================
SVG_FILTER = """
<svg style="width:0; height:0; position:absolute; z-index:-1;" aria-hidden="true" focusable="false" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="glass-filter-_r_b_" color-interpolation-filters="sRGB" x="0%" y="0%" width="100%" height="100%">
       <feImage x="0" y="0" width="100%" height="100%" preserveAspectRatio="none" result="map" href="data:image/png;base64,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"/>
       <feDisplacementMap in="SourceGraphic" in2="map" id="redchannel" result="dispRed" scale="-20" xChannelSelector="R" yChannelSelector="G"></feDisplacementMap>
       <feColorMatrix in="dispRed" type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0" result="red"></feColorMatrix>
       <feDisplacementMap in="SourceGraphic" in2="map" id="greenchannel" result="dispGreen" scale="-24" xChannelSelector="R" yChannelSelector="G"></feDisplacementMap>
       <feColorMatrix in="dispGreen" type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0" result="green"></feColorMatrix>
       <feDisplacementMap in="SourceGraphic" in2="map" id="bluechannel" result="dispBlue" scale="-28" xChannelSelector="R" yChannelSelector="G"></feDisplacementMap>
       <feColorMatrix in="dispBlue" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0" result="blue"></feColorMatrix>
       <feBlend in="red" in2="green" mode="screen" result="rg"></feBlend>
       <feBlend in="rg" in2="blue" mode="screen" result="output"></feBlend>
       <feGaussianBlur in="output" stdDeviation="3"></feGaussianBlur>
    </filter>
  </defs>
</svg>
"""
st.markdown(SVG_FILTER, unsafe_allow_html=True)

# ============================================================
# 3. CUSTOM CSS — TEMA "LIQUID GLASS" RESPONSIVE
# ============================================================
CUSTOM_CSS = f"""
<style>
:root{{
    --text: #1c1c1e;
    --muted: #3a3a40; /* Dibuat sedikit lebih gelap agar terbaca */
    --blue: #0a84ff;
    --blue-soft: rgba(10,132,255,0.15);
    --green: #21963c;
    --green-soft: rgba(48,179,80,0.14);
    --red: #d93025;
    --red-soft: rgba(255,69,58,0.14);
    
    --filter-id: url(#glass-filter-_r_b_); 
    --glass-saturation: 1.5; 
}}

html, body, [data-testid="stAppViewContainer"]{{
    background-image: url("data:image/jpeg;base64,{img_base64}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    color: var(--text);
}}

[data-testid="stHeader"]{{ background: transparent; }}
[data-testid="stSidebar"]{{
    background: rgba(255, 255, 255, 0.4); 
    backdrop-filter: var(--filter-id) saturate(var(--glass-saturation)) !important;
    -webkit-backdrop-filter: var(--filter-id) saturate(var(--glass-saturation)) !important;
    border-right: 1px solid rgba(255,255,255,0.7);
}}
.block-container{{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }}

/* ---- TEMA LIQUID GLASS UNTUK PANEL BESAR, TOPNAV, & METRIK ---- */
[data-testid="stVerticalBlockBorderWrapper"], .topnav, .metric-card {{
    background: rgba(255, 255, 255, 0.55) !important; 
    backdrop-filter: var(--filter-id) saturate(var(--glass-saturation)) !important;
    -webkit-backdrop-filter: var(--filter-id) saturate(var(--glass-saturation)) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    border-top: 1px solid rgba(255, 255, 255, 1) !important; /* Efek pantulan cahaya tajam di atas */
    border-left: 1px solid rgba(255, 255, 255, 1) !important; /* Efek pantulan cahaya tajam di kiri */
    box-shadow: inset 0 0 4px 2px rgba(255,255,255,0.4), inset 0 0 12px 4px rgba(255,255,255,0.15), 0 16px 32px rgba(0, 0, 0, 0.12) !important;
    border-radius: 26px !important;
    overflow: visible !important;
}}

/* Memaksa elemen lapisan dalam dari Streamlit menjadi transparan agar kaca tembus pandang */
[data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {{
    background: transparent !important;
    background-color: transparent !important;
}}

/* Merapikan padding bawaan container agar tidak terlalu mepet */
[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 1rem !important; 
}}

/* ---- Top nav bar ---- */
.topnav{{
    display:flex; align-items:center; justify-content:space-between;
    padding: 14px 22px; margin-bottom: 22px;
}}
.topnav .brand{{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.05rem; color: var(--text); }}
.topnav .brand-badge{{
    width:34px; height:34px; border-radius:12px; background: linear-gradient(135deg,var(--blue),#5ec8ff);
    display:flex; align-items:center; justify-content:center; font-size:18px;
    box-shadow: 0 4px 10px rgba(10,132,255,0.35);
}}
.topnav .menu{{ display:flex; gap:6px; }}
.topnav .menu span{{ padding:8px 16px; border-radius:14px; font-size:0.86rem; color: var(--muted); }}
.topnav .menu span.active{{ background: var(--blue-soft); color: var(--blue); font-weight:600; }}
.topnav .user{{ display:flex; align-items:center; gap:10px; }}
.topnav .avatar{{
    width:36px; height:36px; border-radius:50%; background: linear-gradient(135deg,var(--blue),#8fd3ff);
    display:flex; align-items:center; justify-content:center; font-weight:700; color:#ffffff;
}}
.topnav .user-meta{{ font-size:0.78rem; color: var(--muted); line-height:1.1; }}
.topnav .user-meta b{{ color: var(--text); font-size:0.86rem; }}

/* ---- Section title ---- */
.section-title{{ display:flex; align-items:center; justify-content:space-between; margin: 4px 0 16px 0; }}
.section-title h2{{ font-size:1.4rem; margin:0; color: var(--text); }}
.pill-tabs span{{
    padding:8px 18px; border-radius:14px; font-size:0.82rem; margin-left:8px; color: var(--muted);
    border:1px solid rgba(0,0,0,0.06); background: rgba(255,255,255,0.5);
}}
.pill-tabs span.active{{ background: var(--blue); color:#ffffff; font-weight:700; border:none; box-shadow:0 4px 12px rgba(10,132,255,0.35); }}

/* ---- Metric cards ---- */
.metric-card{{ padding: 18px 20px; height: 100%; }}
.metric-card .label{{ color: var(--muted); font-size:0.82rem; margin-bottom:10px; font-weight: 600; }}
.metric-card .value-row{{ display:flex; align-items:baseline; gap:10px; }}
.metric-card .value{{ font-size:1.65rem; font-weight:700; color: var(--text); }}
.badge{{
    display:inline-flex; align-items:center; gap:4px; font-size:0.74rem; font-weight:700;
    padding: 3px 9px; border-radius: 999px;
}}
.badge-green{{ background: var(--green-soft); color: var(--green); }}
.badge-red{{ background: var(--red-soft); color: var(--red); }}

h4 {{ font-size: 1.1rem; margin-bottom: 15px; color: var(--text); font-weight:700; }}

/* ---- Streamlit widget restyle (glass pill) ---- */
div[data-baseweb="select"] > div{{
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(255,255,255,0.8) !important;
    border-radius: 12px !important; color: var(--text) !important;
}}
ul[data-baseweb="menu"] {{
    background: rgba(255,255,255,0.95) !important;
}}
.stSlider [data-baseweb="slider"]{{ padding-top: 6px; }}

/* ---- TEMA DARK LIQUID GLASS UNTUK TOMBOL ---- */
.stButton > button {{
    background: rgba(25, 28, 35, 0.75) !important; 
    backdrop-filter: var(--filter-id) saturate(1.5) !important;
    -webkit-backdrop-filter: var(--filter-id) saturate(1.5) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-top: 1px solid rgba(255,255,255,0.4) !important;
    border-left: 1px solid rgba(255,255,255,0.4) !important;
    border-radius: 999px !important; 
    font-weight: 700 !important; 
    padding: 10px 0 !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3), inset 0 1px 2px rgba(255,255,255,0.2) !important;
    transition: all 0.3s ease;
}}

/* Memaksa semua elemen teks di dalam tombol berubah warna */
.stButton > button * {{
    color: #ffffff !important; 
}}

.stButton > button:hover {{ 
    background: rgba(45, 50, 60, 0.85) !important; 
    box-shadow: 0 10px 28px rgba(0,0,0,0.5), inset 0 1px 2px rgba(255,255,255,0.4) !important;
}}

.stButton > button:hover * {{
    color: #ffffff !important; 
}}

.stButton > button:hover * {{
    color: #ffffff !important; 
}}

/* Styling tombol browse di file uploader */
[data-testid="stFileUploader"] button {{
    background: rgba(25, 28, 35, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}}

/* Mengubah warna font teks tombol browse */
[data-testid="stFileUploader"] button * {{
    color: #f2f0ef !important; /* Ganti dengan warna yang diinginkan */
}}

/* Warna font saat tombol di-hover */
[data-testid="stFileUploader"] button:hover * {{
    color: #ffffff !important;
}}

/* Mengubah font label petunjuk dropzone (Drag and drop file here / Limit size) */
[data-testid="stFileUploaderDropzone"] * {{
    color: #1c1c1e !important;
    font-weight: 500 !important;
}}

[data-testid="stChatInput"] {{
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: var(--filter-id) saturate(var(--glass-saturation)) !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.1) !important;
    border-radius: 20px !important;
}}
[data-testid="stChatInput"] textarea {{ color: var(--text) !important; font-weight:500; }}

hr{{ border-color: rgba(0,0,0,0.15) !important; }}
h1, h2, h3, h5, p, label, span{{ color: var(--text); }}
.stCaption, [data-testid="stCaptionContainer"]{{ color: var(--muted) !important; font-weight: 600; }}
[data-testid="stMetricValue"]{{ color: var(--text); }}
.stChatMessage{{
    background: rgba(255,255,255,0.7) !important; 
    backdrop-filter: var(--filter-id);
    border-radius: 18px !important; border:1px solid rgba(255,255,255,0.9) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge(value):
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
# PANEL KIRI
# ==========================================
with col_kiri:
    st.markdown("""
    <div class="section-title">
        <h2>Simulasi Model Machine Learning</h2>
        <div class="pill-tabs"><span class="active">Dashboard</span><span>Portofolio</span><span>Statistik</span></div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # MENGGUNAKAN CONTAINER KOSONG UNTUK MEMBALIK URUTAN VISUAL
    # --------------------------------------------------------
    wadah_grafik_dan_metrik = st.container()
    wadah_pengaturan = st.container()

# ==========================================
# 1. PENGATURAN MODEL (Logika diletakkan lebih dulu, tapi visualnya di bawah)
# ==========================================
with wadah_pengaturan:
    # Membungkus widget di dalam container resmi Streamlit agar CSS bekerja
    with st.container(border=True):
        st.markdown('<h4>Pengaturan Model ⚙️</h4>', unsafe_allow_html=True)
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

        batas_waktu = '2023-01-01' # Dimundurkan sedikit agar grafiknya lebih panjang
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

        # Grafik Utama
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_test.index, y=df_test['Total_Uang_Pasif'],
            mode='lines', name='Beli & Diam (Pasif)',
            line=dict(color='#9a9aa0', dash='dash', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_test.index, y=df_test['Total_Uang_Strategi'],
            mode='lines', name=f'Strategi ML: {pilihan_model}',
            line=dict(color='#0a84ff', width=3),
            fill='tozeroy', fillcolor='rgba(10,132,255,0.08)'
        ))
        fig.update_layout(
            title=f"Kinerja Model {pilihan_model} ({ticker_saham}) vs Beli & Diam",
            xaxis_title="Tanggal", yaxis_title="Total Nilai Uang (Rp)",
            template="plotly_white", font=dict(color="#1c1c1e"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"), yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            hovermode="x unified", legend=dict(orientation="h", y=1.1),
            margin=dict(l=10, r=10, t=60, b=10),
        )

        return_pasif = (df_test['Total_Uang_Pasif'].iloc[-1] - modal_awal) / modal_awal * 100
        return_strategi = (df_test['Total_Uang_Strategi'].iloc[-1] - modal_awal) / modal_awal * 100
        rata_untung = df_test['Strategy_Return'].mean()
        volatilitas = df_test['Strategy_Return'].std()
        sharpe = (rata_untung / volatilitas * np.sqrt(252)) if volatilitas != 0 else 0

        # Donut Chart
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
            marker=dict(colors=['#0a84ff', '#30b350', '#c7c7cc'],
                        line=dict(color='rgba(255,255,255,0.8)', width=2)),
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


# ==========================================
# 2. GRAFIK & METRIK (Akan tampil di atas pengaturan)
# ==========================================
with wadah_grafik_dan_metrik:
    if 'metrik_terakhir' in st.session_state:
        m = st.session_state['metrik_terakhir']
        
        # Metrik Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card("Return Pasif", f"{m['return_pasif']:.1f}%", m['return_pasif']), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Return Strategi", f"{m['return_strategi']:.1f}%", m['return_strategi']), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Sharpe Ratio", f"{m['sharpe']:.2f}", m['sharpe']), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("Modal Akhir", f"Rp {m['modal_akhir']:,.0f}", m['return_strategi']), unsafe_allow_html=True)
        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

        # Panel Chart Utama & Donut Chart
        chart_col, donut_col = st.columns([7, 3])
        with chart_col:
            # Dibungkus st.container agar mendapatkan liquid glass dari CSS kita
            with st.container(border=True):
                st.markdown('<h4>Kinerja Strategi ···</h4>', unsafe_allow_html=True)
                st.plotly_chart(st.session_state['grafik_terakhir'], use_container_width=True)

        with donut_col:
            with st.container(border=True):
                st.markdown('<h4>Kontribusi Fitur</h4>', unsafe_allow_html=True)
                st.plotly_chart(st.session_state['donut_terakhir'], use_container_width=True)
                colors = ['#0a84ff', '#30b350', '#8e8e93']
                for (nama, pct), warna in zip(st.session_state['fitur_terakhir'], colors):
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;font-size:0.9rem;margin-bottom:6px; font-weight: 500;'>"
                        f"<span>● {nama}</span><span style='color:{warna};font-weight:800'>{pct:.1f}%</span></div>",
                        unsafe_allow_html=True
                    )
    else:
        # Jika belum ada simulasi, tampilkan pesan informatif yang juga dilindungi panel kaca
        with st.container(border=True):
            st.info("👈 Silakan atur konfigurasi Anda dan tekan tombol 'Jalankan Model' di bawah untuk melihat grafik simulasi.")


# ==========================================
# PANEL KANAN: AI CHATBOT PEMANTAU
# ==========================================
with col_kanan:
    # Membungkus bot chat agar memiliki border dan liquid glass
    with st.container(border=True):
        st.markdown("<h4>🤖 AI Assistant</h4>", unsafe_allow_html=True)
        st.markdown("<div style='color: var(--muted); font-size:0.85rem; margin-bottom:15px;'>Asisten ini memantau performa modelmu.</div>", unsafe_allow_html=True)

        wadah_chat = st.container(height=650)

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

                konteks_sistem = "Kamu adalah asisten Quantitative Analyst profesional. Jawablah dalam bahasa Indonesia dengan singkat dan padat. "

                if 'metrik_terakhir' in st.session_state:
                    data = st.session_state['metrik_terakhir']
                    konteks_sistem += (
                        f"Saat ini, pengguna sedang menguji algoritma {data['model']} pada saham {data['ticker']}. "
                        f"Hasil simulasinya: Total Untung Strategi adalah {data['return_strategi']:.2f}%, "
                        f"Sharpe Ratio {data['sharpe']:.2f}. Strategi pasif menghasilkan {data['return_pasif']:.2f}%. "
                    )
                else:
                    konteks_sistem += "Pengguna belum menjalankan model apa pun."

                pesan_untuk_ai = [{"role": "system", "content": konteks_sistem}]
                pesan_untuk_ai.extend(st.session_state.pesan_chat)

                with st.chat_message("assistant"):
                    wadah_balasan = st.empty()
                    wadah_balasan.write("Sedang menganalisis...")
                    try:
                        respons = ollama.chat(model='llama3.2', messages=pesan_untuk_ai)
                        balasan_ai = respons['message']['content']
                        wadah_balasan.write(balasan_ai)
                        st.session_state.pesan_chat.append({"role": "assistant", "content": balasan_ai})
                    except Exception:
                        wadah_balasan.error("Gagal menghubungi AI lokal. Pastikan aplikasi Ollama berjalan.")