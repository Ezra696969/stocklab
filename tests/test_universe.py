"""Uji validasi lintas saham.

Tidak ada yang menyentuh jaringan: aset dibuat sintetis, dan sebagiannya sengaja
dibuat berkorelasi tinggi untuk menguji bahwa laporan menyadarinya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocklab.validation import Experiment, cross_section
from tests.conftest import make_ohlcv


def make_builder(*, correlated: bool = False, broken: tuple[str, ...] = ()):
    """Bangun fungsi pembentuk eksperimen untuk aset sintetis.

    Kalau ``correlated``, seluruh aset berbagi faktor pasar yang sama dan hanya
    berbeda pada deraunya — meniru saham dari satu bursa.
    """
    market = make_ohlcv(n=500, seed=1)["close"].pct_change().fillna(0.0)

    def build(asset: str) -> Experiment:
        if asset in broken:
            raise ValueError(f"data untuk {asset} tidak tersedia")

        seed = abs(hash(asset)) % 10_000
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.004, len(market))
        steps = (0.85 * market.to_numpy() + noise) if correlated else rng.normal(
            0.0003, 0.012, len(market)
        )
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(steps)), index=market.index, name="close"
        )

        features = pd.DataFrame(
            {"momentum": prices.pct_change(5)}, index=prices.index
        )
        target = (prices.shift(-1) > prices).astype(float)

        def signals(X, y, params):
            return (X["momentum"] > params.get("cutoff", 0.0)).astype(float)

        return Experiment(
            prices=prices,
            features=features,
            target=target,
            build_signals=signals,
            backtest_kwargs={"cost_bps": 10.0},
        )

    return build


ASSETS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


class TestCrossSection:
    def test_runs_every_asset(self):
        report = cross_section(make_builder(), ASSETS)

        assert report.n_assets == len(ASSETS)
        assert list(report.table.index) == ASSETS

    def test_reports_the_spread_not_the_winner(self):
        """Melaporkan aset terbaik adalah multiple testing yang paling mudah
        dilakukan tanpa sadar, jadi tidak ada metode yang mengembalikannya."""
        report = cross_section(make_builder(), ASSETS)

        assert not hasattr(report, "best")
        assert report.worst_sharpe <= report.median_sharpe
        assert 0.0 <= report.win_rate <= 1.0

    def test_nominal_and_risk_adjusted_wins_are_separate(self):
        report = cross_section(make_builder(), ASSETS)

        assert 0.0 <= report.win_rate <= 1.0
        assert 0.0 <= report.sharpe_win_rate <= 1.0

    def test_a_failing_asset_does_not_stop_the_study(self):
        report = cross_section(make_builder(broken=("CCC",)), ASSETS)

        assert report.n_assets == len(ASSETS)
        assert report.n_failed == 1
        assert "tidak tersedia" in report.table.loc["CCC", "error"]

    def test_failed_assets_are_excluded_from_the_statistics(self):
        clean = cross_section(make_builder(), ASSETS)
        partial = cross_section(make_builder(broken=("CCC",)), ASSETS)

        assert not np.isnan(partial.median_sharpe)
        assert partial.n_failed == 1 and clean.n_failed == 0

    def test_empty_asset_list_is_refused(self):
        with pytest.raises(ValueError, match="kosong"):
            cross_section(make_builder(), [])


class TestCorrelation:
    """Sepuluh saham dari satu bursa bukan sepuluh bukti."""

    def test_independent_assets_have_low_correlation(self):
        report = cross_section(make_builder(correlated=False), ASSETS)
        assert report.mean_correlation < 0.3

    def test_shared_market_factor_is_detected(self):
        report = cross_section(make_builder(correlated=True), ASSETS)
        assert report.mean_correlation > 0.5

    def test_effective_sample_shrinks_with_correlation(self):
        independent = cross_section(make_builder(correlated=False), ASSETS)
        together = cross_section(make_builder(correlated=True), ASSETS)

        assert together.effective_sample_size < independent.effective_sample_size
        assert together.effective_sample_size < len(ASSETS)

    def test_effective_sample_never_exceeds_the_asset_count(self):
        report = cross_section(make_builder(correlated=False), ASSETS)
        assert report.effective_sample_size <= len(ASSETS) + 1e-9

    def test_single_asset_has_no_correlation_to_report(self):
        report = cross_section(make_builder(), ["AAA"])
        assert np.isnan(report.mean_correlation)


class TestConcerns:
    def test_single_asset_is_called_out(self):
        report = cross_section(make_builder(), ["AAA"])
        found = report.concerns()

        assert any("satu titik data" in note for note in found)

    def test_correlated_universe_is_called_out(self):
        report = cross_section(make_builder(correlated=True), ASSETS)
        found = report.concerns()

        assert any("pengamatan bebas" in note for note in found)

    def test_failures_are_mentioned(self):
        report = cross_section(make_builder(broken=("CCC", "DDD")), ASSETS)
        assert any("gagal dijalankan" in note for note in report.concerns())

    def test_summary_is_serialisable(self):
        import json

        report = cross_section(make_builder(), ASSETS)
        assert "median_sharpe" in json.dumps(report.summary(), default=float)
