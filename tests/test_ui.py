"""Uji lapisan aplikasi yang bisa diuji tanpa merender Streamlit.

Alur utama, pembukuan percobaan, dan penyusunan konteks chatbot sengaja hidup di
luar berkas antarmuka. Ketiganya paling mudah salah, dan kode yang hanya bisa
dijalankan lewat antarmuka hanya bisa diuji lewat antarmuka.
"""

from __future__ import annotations

import numpy as np
import pytest

from stocklab.features import FeatureConfig
from stocklab.llm.client import ChatClient, OllamaStatus
from stocklab.llm.context import build_context, fallback_narrative, format_context
from stocklab.ui.pipeline import RunConfig, make_experiment, run_pipeline
from stocklab.ui.state import Trial, TrialLog, session_defaults
from tests.conftest import make_ohlcv


def config(**overrides) -> RunConfig:
    base = {
        "features": (
            FeatureConfig("rsi", {"window": 14}),
            FeatureConfig("roc", {"window": 10}),
        ),
        "model_name": "logistic_regression",
        "train_size": 200,
        "test_size": 50,
    }
    return RunConfig(**{**base, **overrides})


class TestTrialLog:
    @staticmethod
    def trial(sharpe: float = 1.0, **overrides) -> Trial:
        base = {
            "model": "random_forest",
            "params": {"max_depth": 8},
            "features": ("rsi_14",),
            "sharpe": sharpe,
            "equity_ratio": 1.1,
        }
        return Trial(**{**base, **overrides})

    def test_counts_every_run(self):
        log = TrialLog()
        log.record(self.trial())
        log.record(self.trial())

        assert len(log) == 2

    def test_identical_configuration_is_one_search(self):
        """Menekan Jalankan dua kali tanpa mengubah apa pun bukan pencarian baru."""
        log = TrialLog()
        log.record(self.trial())
        log.record(self.trial())

        assert log.n_distinct == 1

    def test_changing_a_parameter_counts_as_a_new_trial(self):
        log = TrialLog()
        log.record(self.trial(params={"max_depth": 8}))
        log.record(self.trial(params={"max_depth": 4}))

        assert log.n_distinct == 2

    def test_changing_features_counts_as_a_new_trial(self):
        log = TrialLog()
        log.record(self.trial(features=("rsi_14",)))
        log.record(self.trial(features=("rsi_14", "roc_10")))

        assert log.n_distinct == 2

    def test_sharpes_are_deduplicated_per_configuration(self):
        """Deflated Sharpe menghitung pencarian, bukan berapa kali tombol ditekan."""
        log = TrialLog()
        log.record(self.trial(sharpe=1.0))
        log.record(self.trial(sharpe=1.0))
        log.record(self.trial(sharpe=0.5, params={"max_depth": 4}))

        assert sorted(log.sharpes()) == [0.5, 1.0]

    def test_failed_runs_are_excluded_from_the_spread(self):
        log = TrialLog()
        log.record(self.trial(sharpe=float("nan")))
        log.record(self.trial(sharpe=1.2, params={"max_depth": 4}))

        assert len(log.sharpes()) == 1

    def test_best_run_is_reachable(self):
        log = TrialLog()
        log.record(self.trial(sharpe=0.4))
        log.record(self.trial(sharpe=1.4, params={"max_depth": 4}))

        assert log.best().sharpe == 1.4

    def test_best_is_none_when_nothing_succeeded(self):
        assert TrialLog().best() is None

    def test_clear_resets_the_count(self):
        log = TrialLog()
        log.record(self.trial())
        log.clear()

        assert len(log) == 0 and log.n_distinct == 0

    def test_session_defaults_provide_every_key_panels_read(self):
        defaults = session_defaults()
        expected = {
            "trials", "messages", "result", "walkforward", "validation",
            "ohlcv", "features", "model_spec", "uploaded_adapter",
        }
        assert expected <= set(defaults)


class TestRunConfig:
    def test_embargo_follows_the_horizon(self):
        """Embargo yang lebih kecil dari horizon membocorkan periode uji."""
        assert config(horizon=1).embargo == 1
        assert config(horizon=5).embargo == 5

    def test_backtest_kwargs_carry_the_cost_settings(self):
        kwargs = config(cost_bps=12.0, slippage_bps=3.0).backtest_kwargs

        assert kwargs["cost_bps"] == 12.0
        assert kwargs["slippage_bps"] == 3.0
        assert kwargs["execution_lag"] == 1


class TestPipeline:
    def test_runs_end_to_end(self):
        result = run_pipeline(make_ohlcv(n=600), config())

        assert result.walkforward.n_fitted > 0
        assert result.backtest.equity.index.equals(result.ohlcv.index)
        assert result.metrics["sharpe"] == result.metrics["sharpe"]
        assert result.warmup > 0

    def test_both_curves_start_together_on_the_tradable_window(self):
        """Selama warmup model belum boleh berpendapat sementara pasif sudah
        terisi penuh; memasukkan periode itu bukan membandingkan dua keputusan."""
        result = run_pipeline(make_ohlcv(n=700), config(train_size=252, test_size=21))

        assert result.skipped > 200
        assert result.backtest.equity.iloc[0] == pytest.approx(100_000.0)
        assert result.backtest.benchmark_equity.iloc[0] == pytest.approx(100_000.0)
        assert result.backtest.weights.notna().all()

    def test_trimmed_frames_stay_aligned(self):
        result = run_pipeline(make_ohlcv(n=700), config(train_size=252, test_size=21))

        assert result.features.index.equals(result.ohlcv.index)
        assert result.target.index.equals(result.ohlcv.index)
        assert result.backtest.equity.index.equals(result.ohlcv.index)

    def test_reports_where_the_backtest_really_starts(self):
        result = run_pipeline(make_ohlcv(n=600), config())
        summary = result.walkforward_summary()

        assert summary["n_fitted"] > 0
        assert summary["first_prediction"] is not None
        assert 0.0 < summary["coverage"] < 1.0

    def test_period_summary_describes_the_tradable_window(self):
        """Yang dilaporkan periode yang benar-benar diperdagangkan, bukan data mentah."""
        ohlcv = make_ohlcv(n=600)
        result = run_pipeline(ohlcv, config(cost_bps=8.0, slippage_bps=2.0))
        period = result.period()

        assert period["n_bars"] == len(ohlcv) - result.skipped
        assert period["start"] > str(ohlcv.index[0].date())
        assert period["cost_bps"] == 10.0

    def test_spec_describes_the_chosen_model(self):
        result = run_pipeline(make_ohlcv(n=600), config(model_name="ridge"))

        assert result.spec.name == "ridge"
        assert result.spec.signal_kind == "expected_return"

    def test_no_indicators_is_rejected(self):
        with pytest.raises(ValueError, match="minimal satu indikator"):
            run_pipeline(make_ohlcv(n=600), config(features=()))

    def test_history_too_short_is_explained(self):
        """Pesannya harus memberi tahu apa yang bisa diperbaiki user."""
        with pytest.raises(ValueError, match="Perpanjang rentang"):
            run_pipeline(make_ohlcv(n=150), config(train_size=200, test_size=50))

    def test_experiment_reruns_the_same_path(self):
        """Validasi harus memakai alur yang persis sama dengan yang divalidasi."""
        ohlcv = make_ohlcv(n=600)
        settings = config()

        direct = run_pipeline(ohlcv, settings)
        experiment = make_experiment(ohlcv, settings)
        replayed = experiment.signals({})

        np.testing.assert_allclose(
            direct.walkforward.signals.to_numpy(),
            replayed.to_numpy(),
            equal_nan=True,
        )

    def test_experiment_accepts_parameter_overrides(self):
        ohlcv = make_ohlcv(n=600)
        experiment = make_experiment(ohlcv, config(model_name="random_forest"))

        metrics = experiment.run({"n_estimators": 20, "max_depth": 3})
        assert "sharpe" in metrics


class TestChatContext:
    def test_empty_sections_are_omitted(self):
        """Model harus bisa membedakan nilai nol dari belum dijalankan."""
        context = build_context(model_name="rf")

        assert "metrics" not in context
        assert context["model"]["name"] == "rf"

    def test_nan_becomes_null(self):
        context = build_context(metrics={"sharpe": float("nan"), "cagr": 0.1})

        assert context["metrics"]["sharpe"] is None
        assert context["metrics"]["cagr"] == 0.1

    def test_numpy_values_are_converted(self):
        context = build_context(metrics={"sharpe": np.float64(1.5), "n": np.int64(3)})

        assert isinstance(context["metrics"]["sharpe"], float)
        assert isinstance(context["metrics"]["n"], int)

    def test_prompt_text_labels_the_numbers(self):
        text = format_context(
            build_context(metrics={"sharpe": 1.23, "max_drawdown": -0.18})
        )

        assert "Sharpe: 1.230" in text
        assert "-18.00%" in text

    def test_prompt_text_says_when_nothing_has_run(self):
        assert "Belum ada backtest" in format_context({})

    def test_concerns_reach_the_prompt(self):
        text = format_context(
            build_context(
                metrics={"sharpe": 1.0},
                validation={"concerns": ["Titik impas biaya terlalu dekat."]},
            )
        )
        assert "Titik impas biaya terlalu dekat." in text


class TestFallbackNarrative:
    def test_explains_what_to_do_before_any_run(self):
        assert "Belum ada backtest" in fallback_narrative({})

    def test_compares_against_the_passive_line(self):
        text = fallback_narrative(
            build_context(
                metrics={"sharpe": 1.2, "benchmark_sharpe": 0.4, "equity_ratio": 1.3}
            )
        )

        assert "1.20" in text and "0.40" in text
        assert "mengungguli" in text

    def test_says_plainly_when_the_strategy_lost(self):
        text = fallback_narrative(
            build_context(
                metrics={"sharpe": 0.1, "benchmark_sharpe": 0.9, "equity_ratio": 0.8}
            )
        )
        assert "tertinggal" in text

    def test_repeats_the_concerns(self):
        text = fallback_narrative(
            build_context(
                metrics={"sharpe": 1.0},
                validation={"concerns": ["Turnover terlalu tinggi."]},
            )
        )
        assert "Turnover terlalu tinggi." in text

    def test_clean_validation_is_not_reported_as_a_pass(self):
        text = fallback_narrative(
            build_context(metrics={"sharpe": 1.0}, validation={"concerns": []})
        )
        assert "bukan berarti strategi ini bagus" in text

    def test_heavy_searching_is_mentioned(self):
        text = fallback_narrative(
            build_context(metrics={"sharpe": 1.0}, trials={"n_distinct": 22})
        )
        assert "22 konfigurasi" in text


class TestChatClient:
    def test_missing_package_is_explained_without_jargon(self):
        status = OllamaStatus(installed=False, reachable=False)
        assert "belum terpasang" in status.explain()

    def test_service_down_suggests_the_command(self):
        status = OllamaStatus(installed=True, reachable=False)
        assert "ollama serve" in status.explain()

    def test_no_models_suggests_a_pull(self):
        status = OllamaStatus(installed=True, reachable=True, models=())
        assert "ollama pull" in status.explain()

    def test_offline_client_still_answers(self):
        """Peninjau yang tidak memasang Ollama tetap harus dapat ringkasan."""
        client = ChatClient(status=OllamaStatus(installed=False, reachable=False))
        context = build_context(metrics={"sharpe": 1.1, "benchmark_sharpe": 0.3})

        assert not client.available
        assert "1.10" in client.reply("bagaimana hasilnya?", context)

    def test_preferred_model_is_matched_by_family(self):
        status = OllamaStatus(
            installed=True, reachable=True, models=("llama3.2:3b", "qwen2.5:7b")
        )
        assert ChatClient("llama3.2", status=status).model == "llama3.2:3b"

    def test_falls_back_to_whatever_is_installed(self):
        status = OllamaStatus(installed=True, reachable=True, models=("qwen2.5:7b",))
        assert ChatClient("llama3.2", status=status).model == "qwen2.5:7b"

    def test_greeting_reports_the_offline_reason(self):
        client = ChatClient(status=OllamaStatus(installed=True, reachable=False))
        assert "ollama serve" in client.greeting({})
