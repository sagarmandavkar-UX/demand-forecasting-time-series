import numpy as np
import pandas as pd

from demand_forecasting.models import forecast, rolling_origin_evaluate


def _series(days: int = 450) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=days, freq="D")
    values = 100 + np.arange(days) * 0.2 + 15 * np.sin(2 * np.pi * np.arange(days) / 7)
    return pd.Series(values, index=index)


def test_seasonal_naive_uses_only_last_week() -> None:
    series = _series(60)
    predicted = forecast("seasonal_naive", series, 10)
    np.testing.assert_allclose(predicted.iloc[:7], series.iloc[-7:])


def test_rolling_origin_returns_all_models_and_valid_dates() -> None:
    series = _series()
    metrics, predictions = rolling_origin_evaluate(series, initial=365, horizon=28, step=28)
    assert set(metrics["model"]) == {"last_value", "seasonal_naive", "holt_winters", "autoregression"}
    assert predictions["date"].min() > series.index[364]
    assert (predictions["forecast"] >= 0).all()

