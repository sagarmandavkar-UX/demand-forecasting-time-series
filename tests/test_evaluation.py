import pandas as pd

from demand_forecasting.evaluation import forecast_metrics, interval_coverage


def test_forecast_metrics_are_correct() -> None:
    actual = pd.Series([100, 200, 300])
    predicted = pd.Series([90, 220, 280])
    metrics = forecast_metrics(actual, predicted)
    assert round(metrics["mae"], 2) == 16.67
    assert round(metrics["bias"], 2) == -3.33
    assert round(metrics["wape_pct"], 2) == 8.33


def test_interval_coverage() -> None:
    actual = pd.Series([5, 10, 20, 30])
    lower = pd.Series([0, 9, 10, 35])
    upper = pd.Series([6, 11, 25, 40])
    assert interval_coverage(actual, lower, upper) == 75.0

