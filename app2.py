import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ollama 
import yfinance as yf

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="AI Trading Portofolio", layout="wide")

# 2. Membagi Layar Utama (70% Kiri, 30% Kanan)
col_kiri, col_kanan = st.columns([7, 3])

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
        ticker_saham = st.selectbox(
            "Pilih Saham:", 
            ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "AAPL", "NVDA"]
        )
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
            df = yf.download(ticker_saham, start="2020-01-01")
            
            # Merapikan struktur tabel dari Yahoo Finance 
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 2. Rekayasa Fitur 
            df['Return_1d'] = df['Close'].pct_change()
            
            # Menghitung MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            
            # Menghitung RSI dinamis 
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
            
            # 3. Menyiapkan Data Latih dan Uji
            fitur = ['Return_1d', 'MACD', 'RSI']
            X = df[fitur]
            y = df['Target']
            
            batas_waktu = '2025-01-01'
            X_train = X[X.index < batas_waktu]
            X_test = X[X.index >= batas_waktu]
            y_train = y[y.index < batas_waktu]
            y_test = y[y.index >= batas_waktu]
            
            # 4. Melatih Model
            if pilihan_model == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
            elif pilihan_model == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            else:
                model = RandomForestClassifier(random_state=42)
                
            model.fit(X_train, y_train)
            prediksi = model.predict(X_test)
            
            # 5. Simulasi Keuntungan
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
                line=dict(color='gray', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_test.index, y=df_test['Total_Uang_Strategi'], 
                mode='lines', name=f'Strategi ML: {pilihan_model}', 
                line=dict(color='orange')
            ))
            
            # Menghitung Metrik untuk AI 
            return_pasif = (df_test['Total_Uang_Pasif'].iloc[-1] - modal_awal) / modal_awal * 100
            return_strategi = (df_test['Total_Uang_Strategi'].iloc[-1] - modal_awal) / modal_awal * 100
            
            rata_untung = df_test['Strategy_Return'].mean()
            volatilitas = df_test['Strategy_Return'].std()
            sharpe = (rata_untung / volatilitas * np.sqrt(252)) if volatilitas != 0 else 0
            
            # Menyimpan metrik ke dalam "ingatan" (Session State) Web
            st.session_state['metrik_terakhir'] = {
                'model': pilihan_model,
                'ticker': ticker_saham,
                'return_pasif': return_pasif,
                'return_strategi': return_strategi,
                'sharpe': sharpe
            }

            fig.update_layout(
                title=f"Kinerja Model {pilihan_model} ({ticker_saham}) vs Beli & Diam",
                xaxis_title="Tanggal",
                yaxis_title="Total Nilai Uang (Rp)",
                template="plotly_dark",
                hovermode="x unified"
            )
            
            # Simpan grafik
            st.session_state['grafik_terakhir'] = fig
            
    # === MENAMPILKAN GRAFIK DARI INGATAN ===
    if 'grafik_terakhir' in st.session_state:
        wadah_grafik.plotly_chart(st.session_state['grafik_terakhir'], use_container_width=True)
    else:
        wadah_grafik.info("Tekan tombol 'Jalankan Model' untuk melihat simulasi.")


# ==========================================
# PANEL KANAN: AI CHATBOT PEMANTAU
# ==========================================
with col_kanan:
    st.title("🤖 AI Assistant")
    st.markdown("Asisten ini memantau performa modelmu.")
    
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
                konteks_sistem += f"Saat ini, pengguna sedang menguji algoritma {data['model']} pada saham {data['ticker']}. Hasil simulasinya: Total Untung Strategi adalah {data['return_strategi']:.2f}%, Sharpe Ratio adalah {data['sharpe']:.2f}. Sebagai perbandingan, strategi pasif (Beli & Diam) menghasilkan untung {data['return_pasif']:.2f}%. Gunakan data ini untuk menjawab pertanyaan pengguna jika relevan."
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
                    
                except Exception as e:
                    wadah_balasan.error("Gagal menghubungi AI lokal. Pastikan aplikasi Ollama sedang berjalan di latar belakang.")