"""Forecast evaluation: scoring, comparison, and calibration of densities."""

from .backtest import Backtest, BacktestResult
from .calibration import Calibration, CalibrationResult
from .comparison import (
    ClarkWestTest,
    EncompassingTest,
    ForecastComparison,
    ForecastComparisonResult,
    GiacominiWhiteTest,
    MincerZarnowitzTest,
    clark_west,
    encompassing,
    giacomini_white,
    mincer_zarnowitz,
)
from .conditional import ConditionalForecastResult, conditional_forecast
from .confidence_set import model_confidence_set
from .fan import fan_chart
from .scoring import DensityScore, DensityScoreResult, pinball_loss

__all__ = [
    "Backtest",
    "BacktestResult",
    "Calibration",
    "CalibrationResult",
    "ClarkWestTest",
    "ConditionalForecastResult",
    "DensityScore",
    "DensityScoreResult",
    "EncompassingTest",
    "ForecastComparison",
    "ForecastComparisonResult",
    "GiacominiWhiteTest",
    "MincerZarnowitzTest",
    "clark_west",
    "conditional_forecast",
    "encompassing",
    "fan_chart",
    "giacomini_white",
    "mincer_zarnowitz",
    "model_confidence_set",
    "pinball_loss",
]
