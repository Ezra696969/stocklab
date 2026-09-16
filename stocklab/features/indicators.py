"""Indikator teknikal.

Aturan tunggal yang mengikat seluruh modul ini: **setiap indikator harus
kausal**. Nilai pada bar ``t`` hanya boleh bergantung pada bar ``t`` dan
sebelumnya. Kalau data masa depan ditambahkan di belakang, nilai-nilai yang
sudah terhitung tidak boleh berubah sedikit pun.

Ini terdengar sepele sampai ada yang memakai ``center=True`` pada rolling
window, melakukan ``fillna(method="bfill")``, atau menormalkan dengan rata-rata
seluruh sampel. Ketiganya mengalirkan informasi masa depan ke masa lalu, hasil
backtest jadi mustahil, dan tidak ada satu pun yang tampak salah saat dibaca
sekilas. ``tests/test_indicators.py`` menguji properti ini langsung untuk tiap
indikator, bukan mempercayai peninjauan mata.

Semua fungsi mengembalikan deret yang sejajar dengan masukannya, dengan NaN di
bar warmup. NaN itu jujur: indikator memang belum punya nilai di sana, dan
engine backtest memperlakukannya sebagai posisi kas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "atr",
    "price",
    "bollinger",
    "ema",
    "log_returns",
    "macd",
    "returns",
    "roc",
    "rsi",
    "rolling_volatility",
    "sma",
    "stochastic_k",
]


def price(close: pd.Series, log: bool = True) -> pd.Series:
    """Harga penutupan itu sendiri, opsional dalam skala logaritmik.

    Dipakai model peramal deret waktu, yang menerima riwayat harga alih-alih
    sebaris indikator. Skala log jadi default karena membuat perubahan relatif
    berjarak sama di seluruh rentang harga, yang lebih cocok dengan cara model
    peramal menskalakan masukannya.

    Kausal secara sepele: tiap nilai adalah bar itu sendiri.
    """
    out = np.log(close) if log else close.astype(float)
    return out.rename("price_log" if log else "price")


def returns(close: pd.Series, window: int = 1) -> pd.Series:
    """Return sederhana selama ``window`` bar."""
    _check_window(window)
    return close.pct_change(window).rename(f"return_{window}")


def log_returns(close: pd.Series, window: int = 1) -> pd.Series:
    """Return logaritmik selama ``window`` bar."""
    _check_window(window)
    return np.log(close / close.shift(window)).rename(f"log_return_{window}")


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average."""
    _check_window(window)
    return close.rolling(window).mean().rename(f"sma_{window}")


def ema(close: pd.Series, window: int = 20) -> pd.Series:
    """Exponential moving average.

    ``adjust=False`` dipakai supaya rekursinya benar-benar berjalan maju dan
    nilai lama tidak dihitung ulang saat data baru tiba.
    """
    _check_window(window)
    return close.ewm(span=window, adjust=False).mean().rename(f"ema_{window}")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index dengan pemulusan Wilder.

    Wilder memakai rata-rata bergerak eksponensial dengan ``alpha = 1/window``,
    bukan rata-rata sederhana. Varian rata-rata sederhana (Cutler) bereaksi
    lebih cepat dan memberi angka berbeda; yang dipakai di sini adalah definisi
    aslinya, yang juga dipakai kebanyakan platform charting.
    """
    _check_window(window)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # Tanpa pergerakan turun sama sekali, RS tak hingga dan RSI jenuh di 100.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out.rename(f"rsi_{window}")


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD, garis sinyal, dan histogramnya."""
    _check_window(fast)
    _check_window(slow)
    _check_window(signal)
    if fast >= slow:
        raise ValueError(f"fast ({fast}) harus lebih kecil dari slow ({slow})")

    macd_line = (
        close.ewm(span=fast, adjust=False).mean()
        - close.ewm(span=slow, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": macd_line - signal_line,
        }
    )


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger band beserta posisi relatif harga di dalamnya.

    ``pct_b`` adalah yang biasanya berguna sebagai fitur: 0 di band bawah, 1 di
    band atas, dan tidak terikat skala harga sehingga bisa dibandingkan lintas
    saham.
    """
    _check_window(window)
    if n_std <= 0:
        raise ValueError(f"n_std harus positif, dapat {n_std}")

    middle = close.rolling(window).mean()
    spread = close.rolling(window).std(ddof=0) * n_std
    upper = middle + spread
    lower = middle - spread
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "pct_b": (close - lower) / width,
        }
    )


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average True Range dengan pemulusan Wilder."""
    _check_window(window)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return (
        true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window)
        .mean()
        .rename(f"atr_{window}")
    )


def stochastic_k(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Stochastic %K: posisi penutupan dalam rentang ``window`` bar terakhir."""
    _check_window(window)
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    span = (highest - lowest).replace(0.0, np.nan)
    return (100.0 * (close - lowest) / span).rename(f"stoch_k_{window}")


def rolling_volatility(
    close: pd.Series, window: int = 20, periods_per_year: int = 252
) -> pd.Series:
    """Volatilitas return yang dianualisasi, dihitung bergulir."""
    _check_window(window)
    daily = close.pct_change()
    return (
        daily.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    ).rename(f"volatility_{window}")


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change dalam persen."""
    _check_window(window)
    return (100.0 * close.pct_change(window)).rename(f"roc_{window}")


def _check_window(window: int) -> None:
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(f"window harus bilangan bulat >= 1, dapat {window!r}")
