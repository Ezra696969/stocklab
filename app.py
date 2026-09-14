import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ollama  # <-- TAMBAHKAN BARIS INI
import yfinance as yf

# 1. Konfigurasi Halaman Web
# Layout 'wide' agar layar terpakai penuh dari ujung kiri ke kanan
st.set_page_config(page_title="AI Trading Portofolio", layout="wide")

# 2. Membagi Layar Utama (70% Kiri, 30% Kanan)
col_kiri, col_kanan = st.columns([7, 3])

# ==========================================
# PANEL KIRI: VISUALISASI & KONTROL MODEL
# ==========================================
# ==========================================
# PANEL KIRI: VISUALISASI & KONTROL MODEL
# ==========================================
with col_kiri:
    st.title("📈 Simulasi Model Machine Learning")
    
    # Menyiapkan wadah kosong untuk grafik
    wadah_grafik = st.empty()
    wadah_grafik.info("Tekan tombol 'Jalankan Model' untuk melihat simulasi.")
    
    st.markdown("---") # Garis pembatas
    
    col_model, col_tuning, col_fitur = st.columns(3)
    
    with col_model:
        st.subheader("Pengaturan")
        # --- TAMBAHAN BARU: Dropdown Saham ---
        ticker_saham = st.selectbox(
            "Pilih Saham:", 
            ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "AAPL", "NVDA"]
        )
        # ------------------------------------
        pilihan_model = st.selectbox("Pilih Algoritma:", ["Random Forest", "Logistic Regression"])
        tombol_jalankan = st.button("Jalankan Model 🚀", use_container_width=True)
        
    with col_tuning:
        st.subheader("Tuning")
        window_rsi = st.slider("Periode RSI (Hari)", 7, 30, 14)
        window_sma = st.slider("Periode SMA Pendek", 10, 50, 20)
        
    with col_fitur:
        st.subheader("Fitur")
        st.file_uploader("Import Data Fitur Tambahan (.csv)", type=['csv'])
        st.multiselect("Gunakan Fitur:", ["RSI", "MACD", "Volume Change"], default=["RSI", "MACD"])

    # === LOGIKA SAAT TOMBOL DITEKAN ===
    if tombol_jalankan:
        with st.spinner(f'Mengunduh data {ticker_saham} dan melatih model {pilihan_model}...'):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            # 1. Memuat Data Langsung dari Yahoo Finance
            # Mengambil data dari 1 Januari 2020 hingga hari ini
            df = yf.download(ticker_saham, start="2020-01-01")
            
            # (PENTING) Merapikan struktur tabel dari Yahoo Finance agar cocok dengan kode kita
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 2. Rekayasa Fitur (Sesuai parameter Tuning dari Panel Kiri)
            df['Return_1d'] = df['Close'].pct_change()
            
        with st.spinner(f'Sedang melatih model {pilihan_model}...'):
            # Import library Machine Learning
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            # 1. Memuat Data 
            df = pd.read_csv('BBCA_historical_data.csv', index_col='Date', parse_dates=True)
            
            # 2. Rekayasa Fitur (Sesuai parameter Tuning dari Panel Kiri)
            df['Return_1d'] = df['Close'].pct_change()
            
            # Menghitung MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            
            # Menghitung RSI dinamis berdasarkan slider user (window_rsi)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Target dan Pembersihan
            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # 3. Menyiapkan Data Latih (2020-2024) dan Uji (2025-2026)
            fitur = ['Return_1d', 'MACD', 'RSI']
            X = df[fitur]
            y = df['Target']
            
            batas_waktu = '2025-01-01'
            X_train = X[X.index < batas_waktu]
            X_test = X[X.index >= batas_waktu]
            y_train = y[y.index < batas_waktu]
            y_test = y[y.index >= batas_waktu]
            
            # 4. MELATIH MODEL ASLI BERDASARKAN PILIHAN USER
            if pilihan_model == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
            elif pilihan_model == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            else:
                # Jika XGBoost belum diinstall, kita alihkan sementara ke Random Forest
                model = RandomForestClassifier(random_state=42)
                
            # Proses belajar dan menebak
            model.fit(X_train, y_train)
            prediksi = model.predict(X_test)
            
            # 5. Simulasi Keuntungan Asli
            df_test = df[df.index >= batas_waktu].copy()
            df_test['Prediksi'] = prediksi
            df_test['Strategy_Return'] = df_test['Return_1d'] * df_test['Prediksi'].shift(1)
            df_test.dropna(inplace=True)
            
            modal_awal = 10000000
            df_test['Total_Uang_Pasif'] = modal_awal * (1 + df_test['Return_1d']).cumprod()
            df_test['Total_Uang_Strategi'] = modal_awal * (1 + df_test['Strategy_Return']).cumprod()

            # 6. Membuat Grafik Interaktif Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_test.index, y=df_test['Total_Uang_Pasif'], 
                mode='lines', name='Beli & Diam (Pasif)', 
                line=dict(color='gray')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_test.index, y=df_test['Total_Uang_Strategi'], 
                mode='lines', name=f'Strategi ML: {pilihan_model}', 
                line=dict(color='orange')
            ))
            # --- TAMBAHAN: Menghitung Metrik untuk AI ---
            # Menghitung Total Return (%)
            return_pasif = (df_test['Total_Uang_Pasif'].iloc[-1] - modal_awal) / modal_awal * 100
            return_strategi = (df_test['Total_Uang_Strategi'].iloc[-1] - modal_awal) / modal_awal * 100
            
            # Menghitung Sharpe Ratio Strategi
            rata_untung = df_test['Strategy_Return'].mean()
            volatilitas = df_test['Strategy_Return'].std()
            sharpe = (rata_untung / volatilitas * np.sqrt(252)) if volatilitas != 0 else 0
            
            # Menyimpan metrik ke dalam "ingatan" (Session State) Web
            st.session_state['metrik_terakhir'] = {
                'model': pilihan_model,
                'return_pasif': return_pasif,
                'return_strategi': return_strategi,
                'sharpe': sharpe
            }

            fig.update_layout(
                title=f"Kinerja Model {pilihan_model} vs Beli & Diam",
                xaxis_title="Tanggal",
                yaxis_title="Total Nilai Uang (Rp)",
                template="plotly_dark",
                hovermode="x unified"
            )
            
            # SIMPAN GRAFIK KE DALAM INGATAN WEB
            st.session_state['grafik_terakhir'] = fig
            
    # === MENAMPILKAN GRAFIK DARI INGATAN ===
    # Kode ini diletakkan DI LUAR blok 'if tombol_jalankan:' (sejajar dengan tombol_jalankan)
    if 'grafik_terakhir' in st.session_state:
        wadah_grafik.plotly_chart(st.session_state['grafik_terakhir'], use_container_width=True)
    else:
        wadah_grafik.info("Tekan tombol 'Jalankan Model' untuk melihat simulasi.")

# ==========================================
# PANEL KANAN: AI CHATBOT PEMANTAU
# ==========================================
# ==========================================
# PANEL KANAN: AI CHATBOT PEMANTAU
# ==========================================
with col_kanan:
    st.title("🤖 AI Assistant")
    st.markdown("Asisten ini memantau performa modelmu.")
    
    wadah_chat = st.container(height=500)
    
    # Inisialisasi riwayat chat di memori web
    if "pesan_chat" not in st.session_state:
        st.session_state.pesan_chat = [
            {"role": "assistant", "content": "Halo! Jalankan simulasi di sebelah kiri, lalu tanyakan analisisnya kepada saya."}
        ]
        
    # Menampilkan riwayat chat
    with wadah_chat:
        for pesan in st.session_state.pesan_chat:
            with st.chat_message(pesan["role"]):
                st.write(pesan["content"])
                
    # Kolom input untuk user
    # Kolom input untuk user
    prompt_user = st.chat_input("Tanya AI tentang performa model...")
    if prompt_user:
        # 1. Menampilkan pertanyaan user di layar
        st.session_state.pesan_chat.append({"role": "user", "content": prompt_user})
        with wadah_chat:
            with st.chat_message("user"):
                st.write(prompt_user)
                
            # 2. Menyiapkan Konteks (Prompt Engineering) untuk AI
            konteks_sistem = "Kamu adalah asisten Quantitative Analyst profesional yang ahli di bidang Data Science saham. Jawablah dalam bahasa Indonesia dengan jelas, padat, dan analitis. "
            
            # Jika user sudah menjalankan model, suntikkan datanya secara diam-diam ke otak AI
            if 'metrik_terakhir' in st.session_state:
                data = st.session_state['metrik_terakhir']
                konteks_sistem += f"Saat ini, pengguna sedang menguji algoritma {data['model']}. Hasil simulasinya: Total Untung Strategi adalah {data['return_strategi']:.2f}%, Sharpe Ratio adalah {data['sharpe']:.2f}. Sebagai perbandingan, strategi pasif (Beli & Diam) menghasilkan untung {data['return_pasif']:.2f}%. Gunakan data ini untuk menjawab pertanyaan pengguna jika relevan."
            else:
                konteks_sistem += "Pengguna belum menjalankan model apa pun. Sarankan mereka untuk menekan tombol 'Jalankan Model' di panel kiri terlebih dahulu."
                konteks_sistem += f"Saat ini, pengguna sedang menguji algoritma {data['model']} pada saham {ticker_saham}. Hasil simulasinya: Total Untung Strategi adalah {data['return_strategi']:.2f}%, Sharpe Ratio adalah {data['sharpe']:.2f}..."

            # Menggabungkan instruksi sistem dengan riwayat chat agar AI tidak pelupa
            pesan_untuk_ai = [{"role": "system", "content": konteks_sistem}]
            # Kita kirim juga semua percakapan sebelumnya
            pesan_untuk_ai.extend(st.session_state.pesan_chat)
            
            # 3. Memanggil Ollama (Membiarkan AI Berpikir)
            with st.chat_message("assistant"):
                wadah_balasan = st.empty()
                wadah_balasan.write("Sedang menganalisis data...") # Indikator loading
                
                try:
                    # Memanggil model Llama 3.2 lokal
                    respons = ollama.chat(model='llama3.2', messages=pesan_untuk_ai)
                    balasan_ai = respons['message']['content']
                    
                    # Menampilkan balasan AI yang asli
                    wadah_balasan.write(balasan_ai)
                    
                    # Menyimpan balasan AI ke dalam memori web
                    st.session_state.pesan_chat.append({"role": "assistant", "content": balasan_ai})
                    
                except Exception as e:
                    wadah_balasan.error("Gagal menghubungi AI lokal. Pastikan aplikasi Ollama sedang berjalan di latar belakang.")