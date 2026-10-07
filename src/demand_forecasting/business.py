"""Translate forecasts into transparent planning scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def asymmetric_capacity_cost(
    actual: pd.Series,
    forecast: pd.Series,
    under_cost: float = 3.5,
    over_cost: float = 1.0,
) -> dict[str, float]:
    """Score a hypothetical capacity plan with asymmetric unit costs."""
    difference = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    under_units = np.clip(difference, 0, None).sum()
    over_units = np.clip(-difference, 0, None).sum()
    total_cost = under_units * under_cost + over_units * over_cost
    return {
        "under_units": float(under_units),
        "over_units": float(over_units),
        "scenario_cost": float(total_cost),
    }


def planning_table(
    forecast: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    rides_per_shift: int = 1200,
) -> pd.DataFrame:
    """Create daily base and risk-protected operations staffing guidance."""
    result = pd.DataFrame(
        {
            "forecast": forecast.round().astype(int),
            "lower_80": lower.round().astype(int),
            "upper_80": upper.round().astype(int),
        }
    )
    result["base_shifts"] = np.ceil(result["forecast"] / rides_per_shift).astype(int)
    result["protected_shifts"] = np.ceil(result["upper_80"] / rides_per_shift).astype(int)
    result["capacity_buffer"] = result["upper_80"] - result["forecast"]
    result.index.name = "date"
    return result

