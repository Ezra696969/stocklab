"""Registry indikator yang bisa dirender sendiri oleh UI.

Panel tuning (tengah-bawah) tidak pernah menyebut RSI atau SMA secara langsung.
Panel itu membaca registry ini, lalu merender kontrol dari :class:`ParamSpec`
tiap indikator. Menambah indikator baru cukup dengan menambah satu entri di
sini; UI ikut menyesuaikan tanpa disentuh.

Berbeda dengan parameter model yang datang dari model orang lain, rentang di
sini benar-benar diketahui — indikatornya kita sendiri yang tulis — jadi batas
bawah dan atasnya diisi dan UI boleh memakai slider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from stocklab.features import indicators
from stocklab.params import ParamSpec

__all__ = [
    "FEATURES",
    "FeatureConfig",
    "FeatureSpec",
    "build_features",
    "column_names",
    "get_feature",
    "warmup_bars",
]


@dataclass(frozen=True)
class FeatureSpec:
    """Satu indikator beserta cara memanggil dan merendernya."""

    name: str
    fn: Callable[..., pd.Series | pd.DataFrame]
    inputs: tuple[str, ...]
    """Kolom OHLCV yang dibutuhkan, dilewatkan sebagai argumen posisional."""

    params: dict[str, ParamSpec]
    label: str
    description: str
    outputs: tuple[str, ...] = ()
    """Nama kolom keluaran untuk indikator multi-kolom; kosong berarti satu deret."""

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.params.items()}

    def coerce(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validasi parameter user, isi yang tidak disebut dengan default."""
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(
                f"parameter tidak dikenal untuk '{self.name}': {sorted(unknown)}; "
                f"yang tersedia: {sorted(self.params)}"
            )
        resolved = self.defaults()
        for name, value in params.items():
            resolved[name] = self.params[name].coerce(value)
        return resolved


@dataclass(frozen=True)
class FeatureConfig:
    """Permintaan satu indikator dengan parameter tertentu.

    Berupa daftar, bukan mapping, karena indikator yang sama sering dipakai dua
    kali dengan parameter berbeda — SMA 20 melawan SMA 50 adalah strategi
    crossover paling umum.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


def _window(low: int, high: int, default: int, label: str, help_: str) -> ParamSpec:
    return ParamSpec(
        name="window",
        kind="int",
        default=default,
        low=low,
        high=high,
        step=1,
        label=label,
        help=help_,
    )


FEATURES: dict[str, FeatureSpec] = {
    "returns": FeatureSpec(
        name="returns",
        fn=indicators.returns,
        inputs=("close",),
        params={
            "window": _window(1, 60, 1, "Periode return", "Jarak bar untuk mengukur perubahan harga.")
        },
        label="Return",
        description="Perubahan harga relatif selama sejumlah bar.",
    ),
    "log_returns": FeatureSpec(
        name="log_returns",
        fn=indicators.log_returns,
        inputs=("close",),
        params={
            "window": _window(1, 60, 1, "Periode return", "Jarak bar untuk mengukur perubahan harga.")
        },
        label="Log return",
        description="Return logaritmik; lebih stabil untuk model linier.",
    ),
    "sma": FeatureSpec(
        name="sma",
        fn=indicators.sma,
        inputs=("close",),
        params={
            "window": _window(2, 250, 20, "Periode SMA", "Jumlah bar yang dirata-rata.")
        },
        label="SMA",
        description="Rata-rata bergerak sederhana.",
    ),
    "ema": FeatureSpec(
        name="ema",
        fn=indicators.ema,
        inputs=("close",),
        params={
            "window": _window(2, 250, 20, "Periode EMA", "Bar terbaru diberi bobot lebih besar.")
        },
        label="EMA",
        description="Rata-rata bergerak eksponensial.",
    ),
    "rsi": FeatureSpec(
        name="rsi",
        fn=indicators.rsi,
        inputs=("close",),
        params={
            "window": _window(2, 100, 14, "Periode RSI", "Standarnya 14 bar, memakai pemulusan Wilder.")
        },
        label="RSI",
        description="Momentum 0-100; di atas 70 lazim disebut jenuh beli.",
    ),
    "macd": FeatureSpec(
        name="macd",
        fn=indicators.macd,
        inputs=("close",),
        params={
            "fast": ParamSpec("fast", "int", 12, 2, 100, 1, label="EMA cepat"),
            "slow": ParamSpec("slow", "int", 26, 3, 250, 1, label="EMA lambat"),
            "signal": ParamSpec("signal", "int", 9, 2, 100, 1, label="Garis sinyal"),
        },
        label="MACD",
        description="Selisih dua EMA beserta garis sinyal dan histogramnya.",
        outputs=("macd", "signal", "hist"),
    ),
    "bollinger": FeatureSpec(
        name="bollinger",
        fn=indicators.bollinger,
        inputs=("close",),
        params={
            "window": _window(5, 200, 20, "Periode band", "Bar untuk rata-rata dan deviasi."),
            "n_std": ParamSpec(
                "n_std",
                "float",
                2.0,
                0.5,
                4.0,
                0.1,
                label="Lebar band",
                help="Jumlah standar deviasi dari garis tengah.",
            ),
        },
        label="Bollinger",
        description="Band volatilitas; pct_b tidak terikat skala harga.",
        outputs=("upper", "middle", "lower", "pct_b"),
    ),
    "atr": FeatureSpec(
        name="atr",
        fn=indicators.atr,
        inputs=("high", "low", "close"),
        params={
            "window": _window(2, 100, 14, "Periode ATR", "Rentang sesungguhnya yang dimuluskan Wilder.")
        },
        label="ATR",
        description="Ukuran volatilitas dalam satuan harga.",
    ),
    "stochastic_k": FeatureSpec(
        name="stochastic_k",
        fn=indicators.stochastic_k,
        inputs=("high", "low", "close"),
        params={
            "window": _window(2, 100, 14, "Periode %K", "Jendela untuk mencari tertinggi dan terendah.")
        },
        label="Stochastic %K",
        description="Posisi penutupan dalam rentang beberapa bar terakhir.",
    ),
    "rolling_volatility": FeatureSpec(
        name="rolling_volatility",
        fn=indicators.rolling_volatility,
        inputs=("close",),
        params={
            "window": _window(5, 250, 20, "Periode volatilitas", "Bar untuk menghitung deviasi return."),
            "periods_per_year": ParamSpec(
                "periods_per_year",
                "int",
                252,
                1,
                365,
                1,
                label="Periode per tahun",
                help="252 untuk data harian bursa.",
            ),
        },
        label="Volatilitas",
        description="Deviasi return yang dianualisasi.",
    ),
    "roc": FeatureSpec(
        name="roc",
        fn=indicators.roc,
        inputs=("close",),
        params={
            "window": _window(1, 250, 10, "Periode ROC", "Jarak bar untuk mengukur laju perubahan.")
        },
        label="Rate of change",
        description="Perubahan harga dalam persen.",
    ),
}


def get_feature(name: str) -> FeatureSpec:
    """Ambil spec indikator, dengan pesan error yang membantu kalau salah nama."""
    try:
        return FEATURES[name]
    except KeyError:
        raise KeyError(
            f"indikator '{name}' tidak dikenal; yang tersedia: {sorted(FEATURES)}"
        ) from None


def _format(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p")
    return str(value)


def column_names(config: FeatureConfig) -> list[str]:
    """Nama kolom yang akan dihasilkan konfigurasi ini.

    Parameter ikut masuk ke nama kolom, sehingga SMA 20 dan SMA 50 tidak saling
    menimpa saat dipakai bersamaan.
    """
    spec = get_feature(config.name)
    params = spec.coerce(config.params)
    suffix = "_".join(_format(params[name]) for name in spec.params)
    base = f"{spec.name}_{suffix}" if suffix else spec.name
    if not spec.outputs:
        return [base]
    return [f"{base}_{output}" for output in spec.outputs]


def build_features(
    ohlcv: pd.DataFrame, configs: Sequence[FeatureConfig]
) -> pd.DataFrame:
    """Hitung indikator yang diminta menjadi satu DataFrame.

    Hasilnya sejajar dengan ``ohlcv.index`` dan berisi NaN di bar warmup. NaN
    itu sengaja dipertahankan, bukan diisi: mengisinya mundur akan menarik
    informasi masa depan ke masa lalu, dan mengisinya dengan nol memalsukan
    pendapat yang belum dimiliki indikator.

    Raises
    ------
    ValueError
        Kalau kolom masukan yang dibutuhkan indikator tidak ada di ``ohlcv``.
    """
    if not configs:
        return pd.DataFrame(index=ohlcv.index)

    columns: dict[str, pd.Series] = {}
    for config in configs:
        spec = get_feature(config.name)
        missing = [name for name in spec.inputs if name not in ohlcv.columns]
        if missing:
            raise ValueError(
                f"indikator '{spec.name}' butuh kolom {missing}, "
                f"yang tersedia: {list(ohlcv.columns)}"
            )

        params = spec.coerce(config.params)
        result = spec.fn(*(ohlcv[name] for name in spec.inputs), **params)
        names = column_names(config)

        if isinstance(result, pd.DataFrame):
            for column, output in zip(names, spec.outputs):
                columns[column] = result[output]
        else:
            columns[names[0]] = result

    return pd.DataFrame(columns, index=ohlcv.index)


def warmup_bars(features: pd.DataFrame) -> int:
    """Jumlah bar awal sebelum seluruh indikator punya nilai.

    Dipakai UI untuk memberi tahu kapan backtest sebenarnya dimulai, ketimbang
    membiarkan user mengira modelnya diam saja di awal.
    """
    if features.empty or features.shape[1] == 0:
        return 0
    complete = features.notna().all(axis=1)
    if not complete.any():
        return len(features)
    return int(complete.argmax())
