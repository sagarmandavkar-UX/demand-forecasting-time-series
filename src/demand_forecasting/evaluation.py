"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return scale-aware and business-readable forecast metrics."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    error = predicted_array - actual_array
    absolute = np.abs(error)
    denominator = np.abs(actual_array).sum()
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_pct": float(100 * absolute.sum() / denominator) if denominator else np.nan,
        "bias": float(error.mean()),
    }


def interval_coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Return the percentage of actuals contained by a prediction interval."""
    actual_array = np.asarray(actual, dtype=float)
    return float(100 * np.mean((actual_array >= np.asarray(lower)) & (actual_array <= np.asarray(upper))))

