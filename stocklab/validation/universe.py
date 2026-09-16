"""Validasi lintas saham.

Pemeriksaan terakhir, dan yang paling sering menghabisi strategi yang lolos
semua pemeriksaan sebelumnya. Stabilitas parameter menjawab apakah hasil
bertahan saat slider digeser; stabilitas periode menjawab apakah bertahan di
rentang waktu lain. Yang ini menjawab pertanyaan yang lebih mendasar: apakah
yang ditemukan itu sifat **metodenya**, atau sifat **satu sahamnya**.

Satu saham bukan bukti. Sebuah strategi yang diuji pada BBCA saja dan menang
mungkin menemukan sesuatu yang nyata tentang pasar, atau mungkin menemukan
sesuatu tentang BBCA di periode itu — dan kedua kemungkinan itu menghasilkan
equity curve yang sama meyakinkannya.

Kenapa ini bukan sekadar "ulangi N kali"
----------------------------------------
Dua jebakan yang dijaga modul ini.

**Yang pertama, memilih pemenang.** Menjalankan sepuluh saham lalu melaporkan
yang terbaik adalah bentuk multiple testing yang paling mudah dilakukan tanpa
sadar. Karena itu yang dikembalikan selalu sebarannya — termasuk yang terburuk
— dan tidak ada metode di sini yang mengembalikan "saham terbaik".

**Yang kedua, mengira sepuluh saham berarti sepuluh bukti.** Saham di satu
bursa bergerak bersama; sepuluh saham LQ45 pada dasarnya satu taruhan pada
pasar Indonesia yang diulang sepuluh kali dengan derau berbeda. Korelasi
rata-rata antar aset ikut dilaporkan supaya besarnya masalah itu terlihat,
bukan diasumsikan tidak ada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from stocklab.validation.common import Experiment

__all__ = ["CrossSectionReport", "cross_section"]

#: Membangun eksperimen untuk satu aset. Dibuat sebagai fungsi, bukan daftar
#: Experiment yang sudah jadi, supaya datanya dimuat satu per satu — memuat
#: puluhan saham sekaligus ke memori tidak ada gunanya di sini.
ExperimentBuilder = Callable[[str], Experiment]


@dataclass(frozen=True)
class CrossSectionReport:
    """Hasil alur yang sama dijalankan pada beberapa aset."""

    table: pd.DataFrame
    """Satu baris per aset, berisi metrik dan sebab kegagalan kalau ada."""

    returns: pd.DataFrame
    """Return strategi per aset, untuk mengukur seberapa saling bergantung."""

    @property
    def n_assets(self) -> int:
        return len(self.table)

    @property
    def n_failed(self) -> int:
        return int(self.table["sharpe"].isna().sum())

    @property
    def win_rate(self) -> float:
        """Bagian aset yang mengungguli pasif secara nominal."""
        ratios = self.table["equity_ratio"].dropna()
        return float((ratios > 1.0).mean()) if len(ratios) else float("nan")

    @property
    def sharpe_win_rate(self) -> float:
        """Bagian aset yang mengungguli pasif secara risk-adjusted."""
        usable = self.table[["sharpe", "benchmark_sharpe"]].dropna()
        if usable.empty:
            return float("nan")
        return float((usable["sharpe"] > usable["benchmark_sharpe"]).mean())

    @property
    def median_sharpe(self) -> float:
        values = self.table["sharpe"].dropna()
        return float(values.median()) if len(values) else float("nan")

    @property
    def worst_sharpe(self) -> float:
        values = self.table["sharpe"].dropna()
        return float(values.min()) if len(values) else float("nan")

    @property
    def mean_correlation(self) -> float:
        """Korelasi rata-rata antar return strategi.

        Semakin tinggi, semakin sedikit informasi baru yang dibawa tiap aset
        tambahan. Korelasi 0,8 berarti sepuluh saham lebih mirip dua taruhan
        bebas daripada sepuluh.
        """
        usable = self.returns.dropna(axis=1, how="all")
        if usable.shape[1] < 2:
            return float("nan")
        matrix = usable.corr().to_numpy()
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        upper = upper[~np.isnan(upper)]
        return float(upper.mean()) if len(upper) else float("nan")

    @property
    def effective_sample_size(self) -> float:
        """Berapa aset *bebas* yang setara dengan kumpulan ini.

        Dengan korelasi rata-rata ``r``, ``n`` aset membawa informasi sebanyak
        kira-kira ``n / (1 + (n - 1) * r)`` aset yang saling bebas. Sepuluh
        saham LQ45 yang berkorelasi 0,7 lebih dekat ke dua pengamatan daripada
        sepuluh — dan perbedaan itu menentukan seberapa jauh hasilnya boleh
        digeneralisasi.
        """
        n = int(self.table["sharpe"].notna().sum())
        r = self.mean_correlation
        if n < 2 or np.isnan(r):
            return float(n)
        denominator = 1.0 + (n - 1) * max(r, 0.0)
        return float(n / denominator)

    def summary(self) -> dict[str, float]:
        return {
            "n_assets": float(self.n_assets),
            "n_failed": float(self.n_failed),
            "win_rate": self.win_rate,
            "sharpe_win_rate": self.sharpe_win_rate,
            "median_sharpe": self.median_sharpe,
            "worst_sharpe": self.worst_sharpe,
            "mean_correlation": self.mean_correlation,
            "effective_sample_size": self.effective_sample_size,
        }

    def concerns(self, min_win_rate: float = 0.5) -> list[str]:
        """Hal yang perlu diketahui sebelum menggeneralisasi hasil ini."""
        found: list[str] = []

        if self.n_assets < 2:
            found.append(
                "Hanya satu aset yang diuji. Itu satu titik data, bukan bukti "
                "bahwa metodenya bekerja di luar saham ini."
            )
            return found

        if not np.isnan(self.sharpe_win_rate):
            if self.sharpe_win_rate < min_win_rate:
                found.append(
                    f"Metode ini cuma mengungguli pasif di "
                    f"{self.sharpe_win_rate:.0%} aset yang diuji. Hasil bagus di "
                    f"satu saham kemungkinan besar sifat saham itu, bukan sifat "
                    f"metodenya."
                )

        effective = self.effective_sample_size
        if not np.isnan(effective) and effective < 3.0 and self.n_assets >= 3:
            found.append(
                f"Korelasi antar aset {self.mean_correlation:.2f}, sehingga "
                f"{self.n_assets} saham ini hanya setara sekitar {effective:.1f} "
                f"pengamatan bebas. Menambah saham dari bursa yang sama tidak "
                f"akan banyak menambah bukti."
            )

        if self.n_failed:
            found.append(
                f"{self.n_failed} dari {self.n_assets} aset gagal dijalankan dan "
                f"tidak ikut dihitung."
            )
        return found


def cross_section(
    build: ExperimentBuilder,
    assets: Sequence[str],
    params: Mapping[str, Any] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> CrossSectionReport:
    """Jalankan alur yang sama pada tiap aset dan kumpulkan sebarannya.

    Parameters
    ----------
    build:
        Menghasilkan :class:`~stocklab.validation.common.Experiment` untuk satu
        simbol. Aset yang datanya gagal dimuat dicatat sebagai baris gagal, bukan
        menghentikan seluruh studi.
    assets:
        Simbol yang diuji.
    params:
        Parameter model, sama untuk semua aset — yang diuji di sini asetnya,
        bukan parameternya.
    progress:
        Dipanggil sebelum tiap aset dengan ``(nomor, total, simbol)``. Studi ini
        melatih ulang model untuk tiap aset dan bisa memakan menit; antarmuka
        yang diam selama itu tidak bisa dibedakan dari yang menggantung.
    """
    if not assets:
        raise ValueError("daftar aset kosong")

    rows: list[dict[str, Any]] = []
    series: dict[str, pd.Series] = {}

    for position, asset in enumerate(assets):
        if progress is not None:
            progress(position, len(assets), asset)
        try:
            experiment = build(asset)
            metrics = experiment.run(params)
        except Exception as error:
            rows.append(
                {"asset": asset, "sharpe": float("nan"), "error": str(error)[:200]}
            )
            continue

        rows.append({"asset": asset, **metrics})
        try:
            signals = experiment.signals(params)
            from stocklab.backtest import run_backtest
            from stocklab.validation.common import tradable_slice

            prices, signals = tradable_slice(experiment.prices, signals)
            result = run_backtest(prices, signals, **experiment.backtest_kwargs)
            series[asset] = result.returns
        except Exception:
            # Metriknya sudah didapat; kegagalan di sini hanya berarti korelasi
            # aset ini tidak bisa dihitung.
            pass

    table = pd.DataFrame(rows).set_index("asset")
    returns = pd.DataFrame(series) if series else pd.DataFrame()
    return CrossSectionReport(table=table, returns=returns)
