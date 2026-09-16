"""Pemeriksaan yang berusaha mementahkan hasil backtest.

Empat pertanyaan yang berbeda, dan sebuah strategi bisa lolos tiga lalu gugur
di yang keempat:

- :func:`rotation_test` -- apakah timing model membawa informasi, atau
  keunggulannya cuma efek dari bentuk posisinya?
- :func:`cost_sensitivity` -- pada biaya berapa keunggulan itu habis?
- :func:`parameter_stability` -- apakah hasil bertahan kalau slider digeser?
- :func:`period_stability` -- apakah hasil muncul di rentang waktu lain?
- :func:`cross_section` -- apakah ini sifat metodenya, atau sifat satu sahamnya?

:func:`validate` menjalankan semuanya dan menerjemahkan hasilnya menjadi daftar
kekhawatiran yang bisa dibaca.
"""

from stocklab.validation.common import Experiment, evaluate, rolling_windows
from stocklab.validation.costs import CostSensitivity, cost_sensitivity
from stocklab.validation.report import ValidationReport, validate
from stocklab.validation.rotation import RotationTest, rotation_test
from stocklab.validation.universe import CrossSectionReport, cross_section
from stocklab.validation.stability import (
    StabilityReport,
    parameter_stability,
    period_stability,
)

__all__ = [
    "CostSensitivity",
    "CrossSectionReport",
    "Experiment",
    "RotationTest",
    "StabilityReport",
    "ValidationReport",
    "cost_sensitivity",
    "cross_section",
    "evaluate",
    "parameter_stability",
    "period_stability",
    "rolling_windows",
    "rotation_test",
    "validate",
]
