# Cara menjalankan StockLab (web)

## Menjalankan

Buka terminal (PowerShell/Git Bash), lalu:

```bash
cd E:\Project\STOCKS
python -m streamlit run stocklab_app.py
```

Buka **http://localhost:8501** di browser.

Karena `headless = true` di `.streamlit/config.toml`, browser tidak terbuka otomatis — buka manual.

## Menghentikan

`Ctrl + C` di terminal tempat server jalan.

## Pertama kali / setelah pindah komputer

```bash
cd E:\Project\STOCKS
pip install -e ".[app]"
```

Streamlit sudah terpasang di mesin ini (versi 1.63.0), jadi biasanya langkah ini tidak perlu diulang.

## Chatbot (opsional)

Panel chat memakai Ollama lokal. Kalau mau percakapan penuh:

```bash
ollama pull llama3.2
```

Tanpa Ollama, panel asisten tetap jalan dengan ringkasan metrik dari Python — angkanya sama, hanya tanpa percakapan.

## Kalau bermasalah

| Gejala | Penyebab / solusi |
|---|---|
| `No module named stocklab` | Belum `pip install -e ".[app]"`, atau terminal tidak di folder `STOCKS` |
| `No module named streamlit` | `pip install -e ".[app]"` |
| Port 8501 sudah terpakai | Jalankan dengan port lain: `python -m streamlit run stocklab_app.py --server.port 8502` |
| Data saham gagal diambil | yfinance kadang gagal sementara (koneksi/SSL) — coba lagi, kode sudah retry otomatis 4x |
| Halaman kosong / tidak update | Refresh browser (`F5`), atau stop lalu jalankan ulang server |

## Menjalankan test

```bash
cd E:\Project\STOCKS
pip install -e ".[dev]"
pytest
```

---

*Catatan: EquityLab (`E:\Project\EQUITYLAB`) belum punya web — dashboard-nya baru berupa artifact terpisah. Hanya StockLab yang punya aplikasi Streamlit.*
