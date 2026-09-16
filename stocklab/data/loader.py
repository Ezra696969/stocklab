"""Pemuatan data harga dengan cache lokal.

Semua yang masuk ke sisa paket melewati :func:`normalize_ohlcv`, sehingga
fungsi di hilir hanya perlu menghadapi satu bentuk: kolom huruf kecil
``open/high/low/close/volume``, index waktu naik dan tanpa timezone. Sumbernya
boleh yfinance, CSV, atau apa pun yang ditambahkan nanti.

Index sengaja dibuat tz-naive. Data BEI datang dengan offset ``+07:00``
sementara sumber lain memakai UTC atau tanpa timezone sama sekali; menyamakan
harga penutupan harian lintas timezone tidak ada gunanya dan cuma melahirkan
bug penyelarasan yang sulit dilihat.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

__all__ = [
    "DEFAULT_END",
    "DEFAULT_START",
    "OHLCV_COLUMNS",
    "load_csv",
    "load_ohlcv",
    "normalize_ohlcv",
]

#: Rentang default aplikasi (PLAN.md sec. 2).
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

#: Cache parquet, relatif terhadap direktori kerja. Sudah masuk .gitignore.
DEFAULT_CACHE_DIR = Path("cache")

# Nama kolom yang lazim ditemui, dipetakan ke bentuk kanonik.
_COLUMN_ALIASES = {
    "adj close": "close",
    "adjclose": "close",
    "adj_close": "close",
    "last": "close",
    "price": "close",
    "vol": "volume",
}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rapikan DataFrame harga mentah menjadi bentuk kanonik.

    Menangani kolom MultiIndex dari yfinance, nama kolom yang beragam, index
    ber-timezone, urutan acak, dan tanggal ganda.

    Raises
    ------
    ValueError
        Kalau kolom ``close`` tidak ada. Kolom lain boleh hilang — banyak
        sumber hanya menyediakan harga penutupan — tapi tanpa ``close`` tidak
        ada yang bisa di-backtest.
    """
    out = df.copy()

    # yfinance mengembalikan kolom MultiIndex (field, ticker) bahkan untuk satu
    # ticker; ambil level pertama.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.rename(columns=_COLUMN_ALIASES)
    # Alias bisa menghasilkan nama kembar (close dan adj close); simpan yang pertama.
    out = out.loc[:, ~out.columns.duplicated()]

    if "close" not in out.columns:
        raise ValueError(
            f"kolom 'close' tidak ditemukan; kolom yang ada: {list(out.columns)}"
        )

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=False)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "date"
    # Buang freq yang mungkin terbawa: round-trip parquet tidak menyimpannya,
    # jadi membiarkannya akan membuat hasil cache-hit berbeda dari cache-miss.
    # Lagipula kalender bursa berlubang karena hari libur, sehingga freq apa pun
    # pada data harga sungguhan adalah klaim yang keliru.
    out.index.freq = None

    keep = [c for c in OHLCV_COLUMNS if c in out.columns]
    out = out[keep]

    out = out[~out.index.duplicated(keep="last")].sort_index()
    # Bar tanpa harga penutupan tidak bisa dipakai; sisanya boleh kosong.
    out = out[out["close"].notna()]
    return out


def load_csv(path: str | Path, *, date_column: str | None = None) -> pd.DataFrame:
    """Muat harga dari CSV dan normalisasi.

    ``date_column`` ditebak dari kolom pertama kalau tidak diberikan, yang benar
    untuk ekspor yfinance maupun kebanyakan sumber lain.
    """
    raw = pd.read_csv(path)
    if date_column is None:
        date_column = raw.columns[0]
    if date_column not in raw.columns:
        raise ValueError(
            f"kolom tanggal '{date_column}' tidak ada; kolom: {list(raw.columns)}"
        )
    raw[date_column] = _parse_dates(raw[date_column])
    raw = raw.set_index(date_column)
    return normalize_ohlcv(raw)


def _parse_dates(values: pd.Series) -> pd.Series:
    """Baca kolom tanggal sambil mempertahankan tanggal kalender setempat.

    Mengubah ke UTC lebih dulu akan merusak data harian: bar BEI bertanda
    ``2021-01-04 00:00:00+07:00`` menjadi ``2021-01-03 17:00`` begitu timezone
    dibuang, sehingga seluruh deret mundur satu hari. Kesalahan seperti ini
    tidak melempar error dan tidak terlihat pada grafik — yang terlihat hanya
    tanggal yang jatuh di akhir pekan.

    Jadi offset aslinya dipertahankan dulu, dan pembuangan timezone di
    :func:`normalize_ohlcv` menyisakan waktu dinding setempat.
    """
    try:
        parsed = pd.to_datetime(values, format="mixed")
    except (ValueError, TypeError):
        # Offset yang bercampur tidak bisa diwakili satu timezone; UTC adalah
        # satu-satunya pilihan yang konsisten.
        return pd.to_datetime(values, utc=True, format="mixed")

    if not isinstance(parsed.dtype, pd.DatetimeTZDtype) and parsed.dtype != "datetime64[ns]":
        return pd.to_datetime(values, utc=True, format="mixed")
    return parsed


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    """Path cache yang stabil dan aman jadi nama file untuk satu permintaan."""
    digest = hashlib.sha1(f"{ticker}|{start}|{end}".encode()).hexdigest()[:12]
    safe = "".join(c if c.isalnum() else "_" for c in ticker)
    return cache_dir / f"{safe}_{digest}.parquet"


#: Berapa kali pengambilan diulang sebelum menyerah, dan jeda awalnya.
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 2.0


def _fetch_yfinance(
    ticker: str, start: str, end: str, attempts: int = FETCH_ATTEMPTS
) -> pd.DataFrame:
    """Ambil data dari yfinance, dengan percobaan ulang.

    Kegagalan di sini sering bersifat sementara — sertifikat SSL yang gagal
    diverifikasi, koneksi yang diputus, pembatasan laju — dan ticker yang sama
    berhasil beberapa detik kemudian. Tanpa percobaan ulang, studi lintas saham
    yang mengambil belasan simbol berturut-turut akan kehilangan sebagian
    asetnya secara acak, lalu melaporkan statistik dari sisanya seolah-olah
    itu seluruh rencananya.

    Dipisah dari :func:`load_ohlcv` agar mudah diganti saat pengujian.
    """
    import time

    import yfinance as yf  # impor lambat: paket inti tidak butuh jaringan

    # end di yfinance eksklusif; ditambah satu hari agar user yang meminta
    # 2025-12-31 benar-benar mendapat bar 2025-12-31.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    last_error: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(FETCH_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end_exclusive,
                auto_adjust=True,
                progress=False,
            )
        except Exception as error:
            last_error = error
            continue
        if data is not None and not data.empty:
            return data

    detail = f" ({type(last_error).__name__}: {last_error})" if last_error else ""
    raise ValueError(
        f"yfinance tidak mengembalikan data untuk '{ticker}' pada rentang "
        f"{start}..{end} setelah {attempts} percobaan{detail}"
    )


def load_ohlcv(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    fetcher=_fetch_yfinance,
) -> pd.DataFrame:
    """Muat harga harian satu ticker, lewat cache kalau ada.

    Parameters
    ----------
    ticker:
        Simbol yang dikenal sumber data, misalnya ``"BBCA.JK"``.
    start, end:
        Rentang tanggal inklusif. Default mengikuti rentang aplikasi.
    cache_dir:
        Tempat menyimpan parquet. ``None`` mematikan cache sepenuhnya.
    refresh:
        Paksa ambil ulang walaupun cache ada.
    fetcher:
        Fungsi pengambil data; dapat diganti saat pengujian agar tidak menyentuh
        jaringan.
    """
    path = None
    if cache_dir is not None:
        path = _cache_path(Path(cache_dir), ticker, start, end)
        if path.exists() and not refresh:
            return normalize_ohlcv(pd.read_parquet(path))

    frame = normalize_ohlcv(fetcher(ticker, start, end))
    # Sumber data tidak selalu menghormati rentang yang diminta.
    frame = frame.loc[str(start) : str(end)]
    if frame.empty:
        raise ValueError(f"tidak ada bar untuk '{ticker}' pada {start}..{end}")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    return frame
