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
    raw[date_column] = pd.to_datetime(raw[date_column], utc=True, format="mixed")
    raw = raw.set_index(date_column)
    return normalize_ohlcv(raw)


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    """Path cache yang stabil dan aman jadi nama file untuk satu permintaan."""
    digest = hashlib.sha1(f"{ticker}|{start}|{end}".encode()).hexdigest()[:12]
    safe = "".join(c if c.isalnum() else "_" for c in ticker)
    return cache_dir / f"{safe}_{digest}.parquet"


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Ambil data dari yfinance. Dipisah agar mudah diganti saat pengujian."""
    import yfinance as yf  # impor lambat: paket inti tidak butuh jaringan

    # end di yfinance eksklusif; ditambah satu hari agar user yang meminta
    # 2025-12-31 benar-benar mendapat bar 2025-12-31.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"yfinance tidak mengembalikan data untuk '{ticker}' "
            f"pada rentang {start}..{end}"
        )
    return data


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
