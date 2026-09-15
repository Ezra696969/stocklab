"""Kontrak model, adapter, dan model zoo bawaan."""

from stocklab.models.base import ModelAdapter, ModelSpec, SignalKind, to_weights
from stocklab.models.registry import MODELS, ModelEntry, build_model, get_model_entry
from stocklab.models.sklearn_adapter import SklearnAdapter, infer_signal_kind
from stocklab.models.target import direction_target, forward_return

__all__ = [
    "MODELS",
    "ModelAdapter",
    "ModelEntry",
    "ModelSpec",
    "SignalKind",
    "SklearnAdapter",
    "build_model",
    "direction_target",
    "forward_return",
    "get_model_entry",
    "infer_signal_kind",
    "to_weights",
]
