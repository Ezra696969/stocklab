"""Uji jalur upload artifact (Tier 2).

Yang diuji di sini bukan cuma "modelnya jalan", tapi juga bahwa berkas yang
seharusnya ditolak memang ditolak sebelum dibuka, dan bahwa hal-hal yang tidak
bisa diverifikasi aplikasi dikatakan apa adanya alih-alih ditebak.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocklab.models import ModelAdapter, inspect_uploaded, load_uploaded_model
from stocklab.models.artifact import (
    ArtifactRejected,
    LoadPolicy,
    check_policy,
    detect_format,
)
from stocklab.models.sandbox import SandboxError

PERMISSIVE = LoadPolicy(allow_pickle=True)


def training_frame(n: int = 200, seed: int = 5):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    X = pd.DataFrame(
        {"rsi_14": rng.uniform(10, 90, n), "roc_10": rng.normal(0, 2, n)},
        index=index,
    )
    y = pd.Series((X["rsi_14"] < 50).astype(float), index=index)
    return X, y


def fitted_estimator():
    from sklearn.linear_model import LogisticRegression

    X, y = training_frame()
    return LogisticRegression(max_iter=500).fit(X, y), X, y


@pytest.fixture(scope="module")
def joblib_artifact(tmp_path_factory) -> Path:
    import joblib

    model, _, _ = fitted_estimator()
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture(scope="module")
def onnx_artifact(tmp_path_factory) -> Path:
    """Ekspor tanpa menyertakan nama fitur — bawaan konverter."""
    from skl2onnx import to_onnx

    model, X, _ = fitted_estimator()
    path = tmp_path_factory.mktemp("artifacts") / "model.onnx"
    onnx_model = to_onnx(model, X.to_numpy(dtype=np.float32)[:1])
    path.write_bytes(onnx_model.SerializeToString())
    return path


@pytest.fixture(scope="module")
def onnx_with_names(tmp_path_factory) -> Path:
    """Ekspor yang menyertakan nama fitur di metadata."""
    from skl2onnx import to_onnx

    model, X, _ = fitted_estimator()
    onnx_model = to_onnx(model, X.to_numpy(dtype=np.float32)[:1])
    entry = onnx_model.metadata_props.add()
    entry.key = "feature_names"
    entry.value = json.dumps(list(X.columns))

    path = tmp_path_factory.mktemp("artifacts") / "named.onnx"
    path.write_bytes(onnx_model.SerializeToString())
    return path


class TestFormatDetection:
    """Format ditentukan dari isi berkas, bukan dari ekstensinya."""

    def test_recognises_onnx(self, onnx_artifact: Path):
        assert detect_format(onnx_artifact) == "onnx"

    def test_recognises_joblib(self, joblib_artifact: Path):
        assert detect_format(joblib_artifact) in ("joblib", "pickle")

    def test_pickle_renamed_as_onnx_is_still_pickle(
        self, joblib_artifact: Path, tmp_path: Path
    ):
        """Ekstensi adalah klaim pengunggah; byte pertama adalah kenyataan."""
        disguised = tmp_path / "sebenarnya_pickle.onnx"
        disguised.write_bytes(joblib_artifact.read_bytes())

        assert detect_format(disguised) in ("joblib", "pickle")

    def test_unknown_content_is_rejected(self, tmp_path: Path):
        path = tmp_path / "catatan.onnx"
        path.write_text("ini cuma teks biasa", encoding="utf-8")

        with pytest.raises(ArtifactRejected, match="tidak dikenali"):
            detect_format(path)


class TestPolicy:
    def test_pickle_is_refused_by_default(self, joblib_artifact: Path):
        """Default menolak format yang menjalankan kode saat dimuat."""
        with pytest.raises(ArtifactRejected, match="menjalankan kodenya sendiri"):
            check_policy(joblib_artifact, LoadPolicy())

    def test_pickle_passes_once_allowed(self, joblib_artifact: Path):
        assert check_policy(joblib_artifact, PERMISSIVE) in ("joblib", "pickle")

    def test_onnx_passes_without_any_opt_in(self, onnx_artifact: Path):
        assert check_policy(onnx_artifact, LoadPolicy()) == "onnx"

    def test_oversized_file_is_refused(self, onnx_artifact: Path):
        tiny = LoadPolicy(max_bytes=10)
        with pytest.raises(ArtifactRejected, match="melebihi batas"):
            check_policy(onnx_artifact, tiny)

    def test_empty_file_is_refused(self, tmp_path: Path):
        path = tmp_path / "kosong.onnx"
        path.write_bytes(b"")
        with pytest.raises(ArtifactRejected, match="kosong"):
            check_policy(path, LoadPolicy())

    def test_missing_file_is_refused(self, tmp_path: Path):
        with pytest.raises(ArtifactRejected, match="tidak ditemukan"):
            check_policy(tmp_path / "hilang.onnx", LoadPolicy())

    def test_risk_wording_matches_the_setting(self):
        """Peringatannya harus menyebut konsekuensi, bukan menenangkan."""
        assert "tidak menjalankan kode" in LoadPolicy().describe_risk()

        risky = PERMISSIVE.describe_risk()
        assert "MENJALANKAN KODE" in risky
        assert "bukan pengaman" in risky


class TestOnnxAdapter:
    def test_satisfies_the_protocol(self, onnx_artifact: Path):
        adapter = load_uploaded_model(onnx_artifact, feature_names=["rsi_14", "roc_10"])
        assert isinstance(adapter, ModelAdapter)

    def test_predicts_the_same_as_the_original_estimator(self, onnx_artifact: Path):
        """Konversi ke ONNX tidak boleh mengubah keputusan model."""
        model, X, _ = fitted_estimator()
        adapter = load_uploaded_model(onnx_artifact, feature_names=list(X.columns))

        expected = (model.predict_proba(X)[:, 1] >= 0.5).astype(float)
        np.testing.assert_allclose(adapter.predict_signal(X).to_numpy(), expected)

    def test_feature_names_from_metadata_are_used(self, onnx_with_names: Path):
        """Kalau nama fitur ikut diekspor, urutannya jadi pasti."""
        spec = load_uploaded_model(onnx_with_names).describe()

        assert spec.feature_names == ("rsi_14", "roc_10")
        assert "tidak tersimpan" not in spec.description

    def test_missing_names_are_admitted_not_guessed(self, onnx_artifact: Path):
        """Urutan pilihan user tidak bisa diverifikasi, dan itu harus dikatakan."""
        spec = load_uploaded_model(
            onnx_artifact, feature_names=["rsi_14", "roc_10"]
        ).describe()

        assert "tidak bisa diverifikasi aplikasi" in spec.description

    def test_wrong_feature_count_is_refused(self, onnx_artifact: Path):
        with pytest.raises(ValueError, match="mengharapkan 2 fitur"):
            load_uploaded_model(onnx_artifact, feature_names=["rsi_14"])

    def test_missing_column_is_refused(self, onnx_artifact: Path):
        _, X, _ = fitted_estimator()
        adapter = load_uploaded_model(onnx_artifact, feature_names=list(X.columns))

        with pytest.raises(ValueError, match="butuh fitur"):
            adapter.predict_signal(X[["rsi_14"]])

    def test_columns_are_reordered_to_the_declared_order(self, onnx_with_names: Path):
        _, X, _ = fitted_estimator()
        adapter = load_uploaded_model(onnx_with_names)

        straight = adapter.predict_signal(X)
        shuffled = adapter.predict_signal(X[["roc_10", "rsi_14"]])
        pd.testing.assert_series_equal(straight, shuffled)

    def test_rows_with_nan_get_no_signal(self, onnx_artifact: Path):
        """Warmup indikator tidak boleh dikirim ke model sebagai angka asal."""
        _, X, _ = fitted_estimator()
        X = X.copy()
        X.iloc[:20, 0] = np.nan

        adapter = load_uploaded_model(onnx_artifact, feature_names=list(X.columns))
        weights = adapter.predict_signal(X)

        assert weights.iloc[:20].isna().all()
        assert weights.iloc[20:].notna().all()

    def test_refuses_to_be_retrained(self, onnx_artifact: Path):
        _, X, y = fitted_estimator()
        adapter = load_uploaded_model(onnx_artifact, feature_names=list(X.columns))

        with pytest.raises(NotImplementedError, match="apa adanya"):
            adapter.fit(X, y)

    def test_is_marked_pretrained(self, onnx_artifact: Path):
        spec = load_uploaded_model(
            onnx_artifact, feature_names=["rsi_14", "roc_10"]
        ).describe()

        assert spec.is_pretrained
        assert spec.leakage_is_unverifiable

    def test_declared_train_end_clears_the_leakage_flag(self, onnx_artifact: Path):
        spec = load_uploaded_model(
            onnx_artifact,
            feature_names=["rsi_14", "roc_10"],
            train_end="2022-12-31",
        ).describe()

        assert not spec.leakage_is_unverifiable


class TestSandboxedAdapter:
    def test_satisfies_the_protocol(self, joblib_artifact: Path):
        adapter = load_uploaded_model(joblib_artifact, policy=PERMISSIVE)
        assert isinstance(adapter, ModelAdapter)

    def test_feature_names_are_recovered_from_the_artifact(self, joblib_artifact: Path):
        """Dilatih dengan DataFrame, jadi kolomnya ikut tersimpan."""
        spec = load_uploaded_model(joblib_artifact, policy=PERMISSIVE).describe()

        assert spec.feature_names == ("rsi_14", "roc_10")

    def test_predicts_the_same_as_the_original_estimator(self, joblib_artifact: Path):
        model, X, _ = fitted_estimator()
        adapter = load_uploaded_model(joblib_artifact, policy=PERMISSIVE)

        expected = (model.predict_proba(X)[:, 1] >= 0.5).astype(float)
        np.testing.assert_allclose(adapter.predict_signal(X).to_numpy(), expected)

    def test_parameters_are_surfaced(self, joblib_artifact: Path):
        spec = load_uploaded_model(joblib_artifact, policy=PERMISSIVE).describe()

        assert "max_iter" in spec.params
        assert "random_state" not in spec.params

    def test_rows_with_nan_get_no_signal(self, joblib_artifact: Path):
        _, X, _ = fitted_estimator()
        X = X.copy()
        X.iloc[:15, 1] = np.nan

        weights = load_uploaded_model(
            joblib_artifact, policy=PERMISSIVE
        ).predict_signal(X)

        assert weights.iloc[:15].isna().all()
        assert weights.iloc[15:].notna().all()

    def test_a_broken_artifact_fails_without_taking_down_the_process(
        self, tmp_path: Path
    ):
        """Inilah yang benar-benar diberikan pemisahan proses: kerusakan terkurung."""
        broken = tmp_path / "rusak.joblib"
        broken.write_bytes(b"\x80\x04" + b"sampah yang bukan pickle sah" * 20)

        with pytest.raises(SandboxError):
            load_uploaded_model(broken, policy=PERMISSIVE)

    def test_timeout_is_enforced(self, joblib_artifact: Path):
        impatient = LoadPolicy(allow_pickle=True, timeout_seconds=0.001)

        with pytest.raises(SandboxError, match="dihentikan|jawaban"):
            load_uploaded_model(joblib_artifact, policy=impatient)


class TestCrossFormatAgreement:
    """Dua jalur pemuatan harus sepakat.

    Kalau ONNX dan pickle memberi jawaban berbeda untuk model yang sama, salah
    satunya salah -- dan tanpa pembandingan ini tidak ada yang akan tahu yang
    mana, karena keduanya sama-sama menghasilkan angka yang tampak wajar.
    """

    def test_both_paths_give_the_same_signal(
        self, onnx_with_names: Path, joblib_artifact: Path
    ):
        _, X, _ = fitted_estimator()

        from_onnx = load_uploaded_model(onnx_with_names).predict_signal(X)
        from_pickle = load_uploaded_model(
            joblib_artifact, policy=PERMISSIVE
        ).predict_signal(X)

        pd.testing.assert_series_equal(from_onnx, from_pickle)

    def test_both_paths_report_the_same_features(
        self, onnx_with_names: Path, joblib_artifact: Path
    ):
        onnx_spec = load_uploaded_model(onnx_with_names).describe()
        pickle_spec = load_uploaded_model(joblib_artifact, policy=PERMISSIVE).describe()

        assert onnx_spec.feature_names == pickle_spec.feature_names
        assert onnx_spec.is_pretrained and pickle_spec.is_pretrained


class TestPretrainedPipeline:
    """Model terlatih tidak boleh dilatih ulang oleh alur aplikasi."""

    def test_pipeline_does_not_refit_an_uploaded_model(self, onnx_with_names: Path):
        from stocklab.features import FeatureConfig
        from stocklab.ui.pipeline import RunConfig, run_pipeline
        from tests.conftest import make_ohlcv

        ohlcv = make_ohlcv(n=400)
        adapter = load_uploaded_model(
            onnx_with_names, feature_names=["rsi_14", "roc_10"]
        )
        config = RunConfig(
            features=(FeatureConfig("rsi", {"window": 14}),
                      FeatureConfig("roc", {"window": 10})),
        )

        result = run_pipeline(ohlcv, config, lambda: adapter)

        assert result.spec.is_pretrained
        assert result.walkforward.n_fitted == 0, "artifact tidak boleh dilatih ulang"
        assert result.backtest.equity.notna().all()

    def test_declared_train_end_pushes_the_backtest_forward(self, onnx_with_names: Path):
        """Bar sebelum train_end adalah data yang pernah dilihat model."""
        from stocklab.features import FeatureConfig
        from stocklab.ui.pipeline import RunConfig, run_pipeline
        from tests.conftest import make_ohlcv

        ohlcv = make_ohlcv(n=400)
        cutoff = ohlcv.index[250]
        config = RunConfig(
            features=(FeatureConfig("rsi", {"window": 14}),
                      FeatureConfig("roc", {"window": 10})),
        )

        adapter = load_uploaded_model(
            onnx_with_names, feature_names=["rsi_14", "roc_10"], train_end=cutoff
        )
        result = run_pipeline(ohlcv, config, lambda: adapter)

        assert result.ohlcv.index[0] >= ohlcv.index[249]


class TestInspect:
    def test_onnx_inspection_reports_shape(self, onnx_artifact: Path):
        info = inspect_uploaded(onnx_artifact)

        assert info.format == "onnx"
        assert info.n_features == 2
        assert not info.executes_code_on_load

    def test_onnx_without_names_says_so(self, onnx_artifact: Path):
        info = inspect_uploaded(onnx_artifact)
        assert any("tidak menyimpan nama fitur" in note for note in info.notes)

    def test_joblib_inspection_reports_the_estimator(self, joblib_artifact: Path):
        info = inspect_uploaded(joblib_artifact, PERMISSIVE)

        assert info.estimator == "LogisticRegression"
        assert info.feature_names == ("rsi_14", "roc_10")
        assert info.executes_code_on_load

    def test_joblib_inspection_repeats_the_risk(self, joblib_artifact: Path):
        info = inspect_uploaded(joblib_artifact, PERMISSIVE)
        assert any("bukan sebagai pengaman" in note for note in info.notes)

    def test_inspection_refuses_before_opening(self, joblib_artifact: Path):
        with pytest.raises(ArtifactRejected):
            inspect_uploaded(joblib_artifact, LoadPolicy())
