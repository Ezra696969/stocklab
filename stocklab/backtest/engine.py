"""Engine backtest vektorisasi.

Invariant arsitektur (PLAN.md sec. 4): engine tidak pernah melihat objek model.
Dia hanya menerima deret bobot target. Konversi dari keluaran mentah model
(probabilitas, kelas, expected return) menjadi bobot adalah tugas adapter,
bukan tugas engine. Itu yang membuat modul ini bisa diuji tanpa melatih apa pun.

Konvensi waktu
--------------
``execution_lag=1`` berarti sinyal yang dihitung dari data bar ``t`` menentukan
bobot yang dipegang selama bar ``t+1``, sehingga return yang diperoleh adalah
``close[t+1] / close[t] - 1``. Ini konvensi ``shift(1)`` yang standar: sinyal
diamati pada penutupan bar ``t`` lalu posisi diambil pada harga penutupan itu.

Parameter ini sengaja muncul eksplisit di signature :func:`run_backtest` dan
tidak disembunyikan sebagai konstanta, karena ini pertanyaan pertama yang
diajukan siapa pun yang membaca hasil backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = ["BacktestResult", "run_backtest", "shift_signals"]

#: Satu basis point sebagai pecahan.
BPS = 1e-4


def shift_signals(signals: np.ndarray, lag: int) -> np.ndarray:
    """Geser ``signals`` maju sebanyak ``lag`` bar, isi bagian awal dengan nol.

    Hasilnya ``out[t] == signals[t - lag]`` untuk ``t >= lag``, dan ``0.0``
    untuk bar sebelum itu (belum ada sinyal yang bisa dieksekusi).
    """
    if lag < 0:
        raise ValueError(f"execution_lag tidak boleh negatif, dapat {lag}")
    out = np.zeros(len(signals), dtype=float)
    if lag == 0:
        out[:] = signals
    elif lag < len(signals):
        out[lag:] = signals[:-lag]
    return out


@dataclass(frozen=True)
class BacktestResult:
    """Hasil satu kali backtest.

    Semua deret berbagi index yang sama dengan ``prices`` masukan, sehingga bisa
    langsung diplot berdampingan tanpa penyelarasan ulang.
    """

    equity: pd.Series
    """Equity curve strategi (garis oranye di UI), mark-to-market per bar."""

    benchmark_equity: pd.Series
    """Equity curve buy & hold (garis putih di UI), modal awal sama."""

    returns: pd.Series
    """Return periodik strategi, sudah bersih dari biaya."""

    benchmark_returns: pd.Series
    """Return periodik benchmark, sudah bersih dari biaya."""

    weights: pd.Series
    """Bobot yang benar-benar dipegang tiap bar (sesudah lag diterapkan)."""

    turnover: pd.Series
    """Perubahan bobot absolut per bar; satu round-trip penuh bernilai 2.0."""

    costs: pd.Series
    """Biaya yang dibebankan per bar, dalam satuan mata uang."""

    params: Mapping[str, Any] = field(default_factory=dict)
    """Parameter run, disimpan untuk reproduktibilitas dan pelaporan."""

    @property
    def total_cost(self) -> float:
        """Total biaya transaksi yang dibayar sepanjang backtest."""
        return float(self.costs.sum())

    def summary(self, **kwargs: Any) -> dict[str, float]:
        """Metrik ringkas; pintasan ke :func:`stocklab.backtest.metrics.summarize`."""
        from stocklab.backtest.metrics import summarize

        return summarize(self, **kwargs)


def _validate_inputs(prices: pd.Series, signals: pd.Series | np.ndarray) -> np.ndarray:
    """Periksa harga dan sinyal, kembalikan sinyal sebagai array float."""
    if not isinstance(prices, pd.Series):
        raise TypeError("prices harus pd.Series yang terindeks waktu")
    if len(prices) < 2:
        raise ValueError("butuh minimal 2 bar harga untuk menghitung return")
    if prices.isna().any():
        raise ValueError(
            "prices mengandung NaN; bersihkan atau forward-fill sebelum backtest"
        )
    if (prices <= 0).any():
        raise ValueError("prices harus positif")

    if isinstance(signals, pd.Series):
        if not signals.index.equals(prices.index):
            raise ValueError(
                "index signals tidak sama dengan index prices; "
                "selaraskan dulu agar tidak terjadi salah pasang bar"
            )
        values = signals.to_numpy(dtype=float)
    else:
        values = np.asarray(signals, dtype=float)

    if values.shape != (len(prices),):
        raise ValueError(
            f"signals harus 1-D sepanjang {len(prices)} bar, dapat {values.shape}"
        )
    # NaN selama warmup indikator adalah hal normal: model belum bisa memberi
    # pendapat, jadi diperlakukan sebagai flat (tidak ada posisi).
    return np.nan_to_num(values, nan=0.0)


def _simulate(
    returns: np.ndarray,
    target_weights: np.ndarray,
    *,
    initial_capital: float,
    cost_frac: float,
    execution_lag: int,
) -> dict[str, np.ndarray]:
    """Jalankan satu simulasi equity curve.

    Dipakai untuk strategi maupun benchmark, sehingga keduanya dikenai perlakuan
    lag dan biaya yang identik dan perbandingannya adil.
    """
    weights = shift_signals(target_weights, execution_lag)
    previous = np.concatenate(([0.0], weights[:-1]))
    turnover = np.abs(weights - previous)

    # Biaya dibebankan pada bar saat bobot berubah, sebagai pecahan dari equity.
    cost_pct = turnover * cost_frac
    net_returns = weights * returns - cost_pct

    equity = initial_capital * np.cumprod(1.0 + net_returns)
    # Biaya dalam mata uang memakai equity di awal bar (sebelum bar ini
    # diterapkan), supaya total_cost bisa direkonsiliasi dengan equity curve.
    equity_before = np.concatenate(([initial_capital], equity[:-1]))
    costs = equity_before * cost_pct

    return {
        "equity": equity,
        "returns": net_returns,
        "weights": weights,
        "turnover": turnover,
        "costs": costs,
    }


def run_backtest(
    prices: pd.Series,
    signals: pd.Series | np.ndarray,
    *,
    initial_capital: float = 100_000.0,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    execution_lag: int = 1,
    max_leverage: float = 1.0,
) -> BacktestResult:
    """Backtest sederet bobot target terhadap sederet harga.

    Parameters
    ----------
    prices:
        Harga penutupan, terindeks waktu, positif dan tanpa NaN.
    signals:
        **Bobot target**, bukan keluaran mentah model. ``1.0`` berarti seluruh
        modal di aset, ``0.0`` seluruhnya kas, ``-1.0`` short penuh. NaN
        diperlakukan sebagai flat. Adapter yang bertanggung jawab menerjemahkan
        probabilitas atau expected return ke skala ini.
    initial_capital:
        Modal awal, sama untuk strategi maupun benchmark.
    cost_bps, slippage_bps:
        Biaya transaksi dan slippage dalam basis point, dibebankan pada
        turnover. Keduanya dijumlahkan; dipisah agar bisa dilaporkan terpisah
        di UI.
    execution_lag:
        Jumlah bar antara sinyal terbentuk dan bobot mulai dipegang. Lihat
        catatan konvensi waktu di docstring modul. Default ``1``.
    max_leverage:
        Batas atas ``abs(bobot)``. Pelanggaran memicu error alih-alih di-clip
        diam-diam, karena bobot di luar rentang hampir selalu menandakan bug di
        adapter, bukan niat user.

    Returns
    -------
    BacktestResult
        Equity curve strategi dan benchmark beserta deret pendukungnya.
    """
    values = _validate_inputs(prices, signals)

    if max_leverage <= 0:
        raise ValueError("max_leverage harus positif")
    if np.any(np.abs(values) > max_leverage + 1e-12):
        worst = float(np.max(np.abs(values)))
        raise ValueError(
            f"bobot {worst:.4g} melebihi max_leverage {max_leverage:.4g}; "
            "adapter harus menormalkan keluaran model sebelum masuk engine"
        )
    if initial_capital <= 0:
        raise ValueError("initial_capital harus positif")
    if cost_bps < 0 or slippage_bps < 0:
        raise ValueError("cost_bps dan slippage_bps tidak boleh negatif")

    returns = prices.pct_change().fillna(0.0).to_numpy(dtype=float)
    cost_frac = (cost_bps + slippage_bps) * BPS

    strategy = _simulate(
        returns,
        values,
        initial_capital=initial_capital,
        cost_frac=cost_frac,
        execution_lag=execution_lag,
    )
    # Benchmark pasif lewat jalur kode yang sama: buy & hold dikenai lag dan
    # biaya masuk yang identik, jadi selisih kedua garis murni berasal dari
    # keputusan model.
    benchmark = _simulate(
        returns,
        np.ones(len(values), dtype=float),
        initial_capital=initial_capital,
        cost_frac=cost_frac,
        execution_lag=execution_lag,
    )

    index = prices.index
    return BacktestResult(
        equity=pd.Series(strategy["equity"], index=index, name="strategy"),
        benchmark_equity=pd.Series(benchmark["equity"], index=index, name="passive"),
        returns=pd.Series(strategy["returns"], index=index, name="strategy_return"),
        benchmark_returns=pd.Series(
            benchmark["returns"], index=index, name="passive_return"
        ),
        weights=pd.Series(strategy["weights"], index=index, name="weight"),
        turnover=pd.Series(strategy["turnover"], index=index, name="turnover"),
        costs=pd.Series(strategy["costs"], index=index, name="cost"),
        params={
            "initial_capital": initial_capital,
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
            "execution_lag": execution_lag,
            "max_leverage": max_leverage,
            "n_bars": len(prices),
            "start": index[0],
            "end": index[-1],
        },
    )
