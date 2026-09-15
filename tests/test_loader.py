"""Uji pemuatan dan normalisasi data.

Tidak ada test di sini yang menyentuh jaringan. Pengambilan data disuntikkan
lewat parameter ``fetcher``, sehingga suite tetap bisa jalan offline dan
hasilnya tidak bergantung pada apa yang sedang dilakukan bursa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocklab.data import load_csv, load_ohlcv, normalize_ohlcv

BBCA_CSV = Path(__file__).resolve().parent.parent / "BBCA_historical_data.csv"


def raw_frame(n: int = 10, tz: str | None = None) -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=n, freq="B", tz=tz)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 110, n),
            "High": np.linspace(101, 111, n),
            "Low": np.linspace(99, 109, n),
            "Close": np.linspace(100.5, 110.5, n),
            "Volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )


class TestNormalize:
    def test_lowercases_and_orders_columns(self):
        result = normalize_ohlcv(raw_frame())
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_flattens_yfinance_multiindex_columns(self):
        frame = raw_frame()
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["BBCA.JK"]])

        result = normalize_ohlcv(frame)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_strips_timezone(self):
        """Data BEI datang dengan +07:00; sumber lain UTC atau tanpa timezone."""
        result = normalize_ohlcv(raw_frame(tz="Asia/Jakarta"))
        assert result.index.tz is None

    def test_adj_close_becomes_close(self):
        frame = raw_frame().drop(columns=["Close"]).rename(columns={"Open": "Adj Close"})
        result = normalize_ohlcv(frame)
        assert "close" in result.columns

    def test_duplicate_close_columns_keep_the_first(self):
        frame = raw_frame()
        frame["Adj Close"] = frame["Close"] * 2
        result = normalize_ohlcv(frame)

        assert list(result.columns).count("close") == 1
        assert result["close"].iloc[0] == pytest.approx(100.5)

    def test_sorts_and_deduplicates_index(self):
        frame = pd.concat([raw_frame(5), raw_frame(5)]).iloc[::-1]
        result = normalize_ohlcv(frame)

        assert result.index.is_monotonic_increasing
        assert not result.index.has_duplicates

    def test_rows_without_close_are_dropped(self):
        frame = raw_frame()
        frame.iloc[3, frame.columns.get_loc("Close")] = np.nan

        result = normalize_ohlcv(frame)
        assert len(result) == len(frame) - 1

    def test_close_only_source_is_accepted(self):
        """Banyak sumber cuma punya penutupan; itu sudah cukup untuk backtest."""
        frame = raw_frame()[["Close"]]
        result = normalize_ohlcv(frame)
        assert list(result.columns) == ["close"]

    def test_missing_close_raises(self):
        frame = raw_frame().drop(columns=["Close"])
        with pytest.raises(ValueError, match="close"):
            normalize_ohlcv(frame)

    def test_index_is_named_date(self):
        assert normalize_ohlcv(raw_frame()).index.name == "date"


class TestLoadCsv:
    def test_reads_repository_bbca_file(self):
        if not BBCA_CSV.exists():
            pytest.skip("data contoh BBCA tidak ada")

        frame = load_csv(BBCA_CSV)

        assert {"open", "high", "low", "close", "volume"} <= set(frame.columns)
        assert frame.index.tz is None
        assert frame.index.is_monotonic_increasing
        assert (frame["close"] > 0).all()

    def test_writes_then_reads_round_trip(self, tmp_path: Path):
        source = tmp_path / "prices.csv"
        raw_frame().rename_axis("Date").to_csv(source)

        frame = load_csv(source)
        assert len(frame) == 10
        assert frame["close"].iloc[0] == pytest.approx(100.5)

    def test_unknown_date_column_raises(self, tmp_path: Path):
        source = tmp_path / "prices.csv"
        raw_frame().rename_axis("Date").to_csv(source)

        with pytest.raises(ValueError, match="kolom tanggal"):
            load_csv(source, date_column="Tanggal")


class CountingFetcher:
    """Pengambil data palsu yang mencatat berapa kali dipanggil."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = 0

    def __call__(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        self.calls += 1
        return self.frame


class TestLoadOhlcv:
    def test_second_call_is_served_from_cache(self, tmp_path: Path):
        fetcher = CountingFetcher(raw_frame(n=40))

        first = load_ohlcv("BBCA.JK", cache_dir=tmp_path, fetcher=fetcher)
        second = load_ohlcv("BBCA.JK", cache_dir=tmp_path, fetcher=fetcher)

        assert fetcher.calls == 1
        pd.testing.assert_frame_equal(first, second)

    def test_refresh_bypasses_cache(self, tmp_path: Path):
        fetcher = CountingFetcher(raw_frame(n=40))

        load_ohlcv("BBCA.JK", cache_dir=tmp_path, fetcher=fetcher)
        load_ohlcv("BBCA.JK", cache_dir=tmp_path, fetcher=fetcher, refresh=True)

        assert fetcher.calls == 2

    def test_cache_key_separates_tickers_and_ranges(self, tmp_path: Path):
        fetcher = CountingFetcher(raw_frame(n=40))

        load_ohlcv("BBCA.JK", cache_dir=tmp_path, fetcher=fetcher)
        load_ohlcv("BBRI.JK", cache_dir=tmp_path, fetcher=fetcher)
        load_ohlcv("BBCA.JK", "2023-01-01", "2023-06-30", cache_dir=tmp_path, fetcher=fetcher)

        assert fetcher.calls == 3

    def test_cache_can_be_disabled(self, tmp_path: Path):
        fetcher = CountingFetcher(raw_frame(n=40))

        load_ohlcv("BBCA.JK", cache_dir=None, fetcher=fetcher)
        load_ohlcv("BBCA.JK", cache_dir=None, fetcher=fetcher)

        assert fetcher.calls == 2
        assert not list(tmp_path.iterdir())

    def test_range_is_trimmed_to_request(self):
        """Sumber data tidak selalu menghormati rentang yang diminta."""
        fetcher = CountingFetcher(raw_frame(n=200))

        frame = load_ohlcv(
            "BBCA.JK", "2023-02-01", "2023-03-31", cache_dir=None, fetcher=fetcher
        )

        assert frame.index.min() >= pd.Timestamp("2023-02-01")
        assert frame.index.max() <= pd.Timestamp("2023-03-31")

    def test_empty_range_raises(self):
        fetcher = CountingFetcher(raw_frame(n=20))

        with pytest.raises(ValueError, match="tidak ada bar"):
            load_ohlcv(
                "BBCA.JK", "2030-01-01", "2030-12-31", cache_dir=None, fetcher=fetcher
            )

    def test_ticker_with_punctuation_is_a_safe_filename(self, tmp_path: Path):
        fetcher = CountingFetcher(raw_frame(n=40))
        load_ohlcv("^JKSE", cache_dir=tmp_path, fetcher=fetcher)

        written = list(tmp_path.glob("*.parquet"))
        assert len(written) == 1
        assert "^" not in written[0].name
