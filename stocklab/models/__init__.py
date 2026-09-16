"""Kontrak model, adapter, model zoo, dan pemuatan artifact user."""

from stocklab.models.artifact import ArtifactInfo, ArtifactRejected, LoadPolicy
from stocklab.models.catalog import CatalogEntry, build_any, catalog
from stocklab.models.forecaster import FORECASTERS, build_forecaster
from stocklab.models.base import ModelAdapter, ModelSpec, SignalKind, to_weights
from stocklab.models.registry import MODELS, ModelEntry, build_model, get_model_entry
from stocklab.models.sklearn_adapter import SklearnAdapter, infer_signal_kind
from stocklab.models.target import direction_target, forward_return
from stocklab.models.upload import inspect_uploaded, load_uploaded_model

__all__ = [
    "FORECASTERS",
    "MODELS",
    "CatalogEntry",
    "ArtifactInfo",
    "ArtifactRejected",
    "LoadPolicy",
    "ModelAdapter",
    "ModelEntry",
    "ModelSpec",
    "SignalKind",
    "SklearnAdapter",
    "build_any",
    "build_forecaster",
    "build_model",
    "catalog",
    "direction_target",
    "forward_return",
    "get_model_entry",
    "infer_signal_kind",
    "inspect_uploaded",
    "load_uploaded_model",
    "to_weights",
]
