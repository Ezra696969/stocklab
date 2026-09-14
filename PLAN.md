# StockLab — Rencana Proyek

Lab backtesting model ML untuk data saham. Analoginya SPSS: user membawa model
dan hipotesisnya, aplikasi mengurus eksperimen, validasi, dan pelaporannya —
tanpa user perlu menulis kode.

**Target pembaca portofolio:** peran ML / quant.
Konsekuensinya, `stocklab/` harus bisa dipakai penuh dari notebook tanpa
menyentuh Streamlit sama sekali. Itu ujian arsitekturnya.

---

## 1. Tesis produk

User bisa melakukan backtesting terhadap model yang ingin mereka ketahui,
tanpa coding. Ini inti proyek, bukan fitur tambahan.

Yang **tidak** dijanjikan: "upload model apa pun dari Hugging Face". Mayoritas
isi HF adalah transformer NLP/vision yang kontrak inputnya token atau piksel,
dan tidak ada cara bermakna mengadaptasi checkpoint BERT jadi sinyal trading.
Janji yang tidak bisa ditepati lebih merugikan daripada fitur yang tidak ada.

Yang dijanjikan sebagai gantinya: kontrak model yang jelas, dengan tiga jalur
masuk.

### Tier 1 — Model zoo bawaan
sklearn / XGBoost / LightGBM. Nol friksi, jalur masuk untuk user yang belum
punya model apa pun.

### Tier 2 — Upload artifact terlatih
Jalur yang paling menjawab tesis. Deteksi parameter berjalan lewat introspeksi
biasa, bukan sihir:

- **sklearn** → `feature_names_in_` (fitur + urutannya) dan `get_params()`
  (hyperparameter untuk dirender jadi slider/dropdown)
- **ONNX** → input signature

Panel parameter terisi otomatis dari sini.

### Tier 3 — Model time-series dari HF, daftar kurasi
Chronos, Lag-Llama, TimesFM, PatchTST. Semuanya menerima deret numerik,
beberapa bisa zero-shot. User memilih dari dropdown, bukan menempel URL
sembarangan.

Ini fase berikutnya, bukan karena kurang penting, tapi karena berdiri di atas
kontrak adapter yang dibangun di Tier 2. Dependensinya berat (torch, unduhan
model).

---

## 2. Layout UI

| Posisi | Isi |
|---|---|
| Kiri-atas | Equity curve: garis oranye (strategi) vs garis putih (pasif / buy & hold) |
| Kanan-atas | Chatbot lokal (Ollama) — evaluator performa model |
| Kiri-bawah | Pemilihan / upload model |
| Tengah-bawah | Tuning variabel indeks (RSI, SMA, MACD, ...) |
| Kanan-bawah | Parameter model (terisi otomatis dari `ModelSpec`) |

Rentang default: awal 2023 – akhir 2025.

**Kedua garis adalah equity curve**, bukan harga saham — mark-to-market harian
dari modal awal yang sama, supaya perbandingannya adil ketika model sering
memegang kas.

---

## 3. Struktur repo

```
stocklab/
├── data/          loader (yfinance + cache parquet), kalender bursa
├── features/      indikator (RSI, SMA, MACD, ATR...), registry fitur
├── models/        adapters/ + builtin/ + registry
├── backtest/      engine.py, metrics.py, walkforward.py
├── llm/           klien Ollama + templating metrik (fallback tanpa LLM)
└── ui/            app.py + komponen panel
tests/
```

Aturan keras: **logika backtest tidak boleh hidup di dalam file Streamlit.**
Reviewer akan membuka `backtest/`, bukan `app.py`. Kalau logikanya bercampur
dengan UI, nilainya hilang berapa pun bagusnya.

---

## 4. Kontrak adapter

Satu protokol, empat implementasi (`Builtin`, `Sklearn`, `ONNX`, `HFTimeSeries`).

```python
class ModelAdapter(Protocol):
    def describe(self) -> ModelSpec: ...
    def predict_signal(self, X: pd.DataFrame) -> np.ndarray: ...
    def fit(self, X, y) -> None: ...   # opsional; pretrained boleh raise
```

`ModelSpec` membawa:

- `feature_names` — fitur yang dibutuhkan model, beserta urutannya
- `params` — hasil introspeksi, mengisi panel kanan-bawah
- `train_end` — lihat §6
- `signal_kind` — `proba` / `class` / `expected_return`

**Invariant arsitektur: engine backtest tidak pernah melihat objek model.**
Dia hanya menerima array sinyal. Itu yang membuat engine bisa dites tanpa
melatih apa pun, dan membuat Tier 3 nanti tinggal menambah satu adapter.

---

## 5. Engine backtest

```python
def run_backtest(prices, signals, *, initial_capital,
                 cost_bps, slippage_bps, execution_lag=1) -> BacktestResult
```

`execution_lag=1` (sinyal di `t` → posisi di `t+1`) sengaja dibuat eksplisit di
signature. Itu pertanyaan pertama setiap reviewer quant; jangan disembunyikan.

Walk-forward untuk model yang dilatih di dalam app: expanding window, latih di
masa lalu, prediksi hanya ke depan. Untuk model pretrained tidak ada refit —
validitasnya bergantung pada `train_end`.

### Invariant test

Empat test yang membuktikan engine benar, semuanya murah:

1. sinyal all-ones + biaya nol ⇒ equity curve **identik** dengan buy & hold
2. sinyal all-zeros ⇒ equity datar di kas awal
3. sinyal digeser satu bar ⇒ hasil ikut bergeser satu bar (membuktikan lag bekerja)
4. N round-trip ⇒ equity berkurang ≈ 2N × cost

### Metrik

Lewati "total return" saja. CAGR, Sharpe, Sortino, max drawdown, Calmar, hit
rate, turnover, exposure — plus relatif terhadap benchmark: alpha, beta,
information ratio. Garis putih sudah jadi benchmark alami, jadi ini gratis.

### Deflated Sharpe

Panel tuning punya konsekuensi yang jarang disadari: setiap kali user menggeser
periode RSI lalu menjalankan ulang, itu satu percobaan tambahan dalam
*multiple testing*. Geser 40 kali, hampir pasti ada satu konfigurasi yang
Sharpe-nya bagus karena kebetulan.

App menghitung jumlah percobaan per sesi dan menampilkan deflated Sharpe di
sebelah Sharpe biasa:

> Anda sudah mencoba 47 konfigurasi; Sharpe 1.8 turun jadi 0.6 setelah dikoreksi.

Datanya sudah ada, tinggal dihitung.

---

## 6. Leakage pada model yang di-upload

Model yang di-upload sudah dilatih di suatu periode, dan app tidak punya cara
mengetahuinya. Kalau training datanya mencakup 2023–2025 lalu di-backtest di
rentang yang sama, hasilnya bocor total dan tidak ada kode yang bisa
mendeteksinya. Ini hazard bawaan semua lab BYO-model.

**Keputusan:** ditangani sebagai aturan yang dikomunikasikan ke user, bukan
dipaksakan.

- Dokumentasikan sebagai rule eksplisit di UI dan README
- Sediakan field opsional `train_end`; kalau diisi, awal backtest di-clamp
  otomatis ke setelah tanggal itu
- Kalau dikosongkan, hasil jadi tanggung jawab user

Field opsional itu sekitar 20 baris dan mengubah pembacaan proyek dari
"tahu masalahnya, lalu melempar ke user" menjadi "tahu masalahnya, dan
menyediakan jalan keluarnya".

---

## 7. Keamanan

Kalau upload adalah inti produk, `joblib.load()` / `pickle.load()` pada file
asing bukan risiko sampingan — itu permukaan serang utama. Deserialisasi pickle
mengeksekusi kode arbitrer.

- Allowlist format; utamakan ONNX / safetensors di atas pickle
- Loading di subprocess terisolasi
- Satu bagian *threat model* di README

Bagian threat model jarang sekali muncul di proyek portofolio ML, dan langsung
terbaca sebagai kematangan.

---

## 8. Chatbot

LLM tidak pernah menghitung. Polanya: Python menghitung seluruh metrik → dikirim
sebagai JSON ke prompt → LLM hanya menerjemahkannya jadi narasi. LLM tidak
pernah diminta mengarang angka.

Ollama lokal dipilih karena privasi dan nol biaya API — tulis alasannya di
README. Tapi reviewer portofolio tidak akan menginstal Ollama, jadi wajib ada
*graceful degradation*: kalau Ollama tidak terdeteksi, panel chat tetap tampil
dengan ringkasan metrik hasil template dari Python. Jangan crash.

---

## 9. Standar rekayasa

Ini portofolio, bukan lomba — tidak ada deadline, standarnya naik.

- Git dengan riwayat commit yang masuk akal (bukan satu commit "initial")
- Struktur paket, bukan `app.py` … `app5.py`
- Test untuk engine backtest (§5)
- README dengan GIF demo + bagian metodologi + threat model
- `pyproject.toml` dengan versi terkunci

**Bagian metodologi di README adalah diferensiator terbesar.** Hampir semua
proyek *stock prediction* di GitHub punya look-ahead bias dan menampilkan equity
curve fantasi. Menulis *"saya pakai walk-forward, eksekusi di t+1, dan
memasukkan biaya transaksi — karena itu hasilnya cuma sedikit di atas buy &
hold, dan itu realistis"* terbaca lebih kuat daripada fitur apa pun.

---

## 10. Urutan pengerjaan

1. **`backtest/` + test-nya.** Fondasi semuanya, paling cepat terlihat hasilnya.
2. `data/` + `features/` — indikator dan registry
3. Kontrak adapter + Tier 1 (model zoo bawaan)
4. Tier 2 (upload sklearn/ONNX + introspeksi parameter)
5. UI Streamlit di atas library yang sudah jadi
6. Chatbot + fallback
7. Tier 3 (HF time-series)

Migrasi dari `app5.py` bersifat bertahap; prototipe lama dipertahankan sebagai
referensi sampai UI baru menyamai fungsinya.
