"""Forecasting models and leakage-safe rolling-origin validation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .evaluation import forecast_metrics

MODEL_NAMES = ("last_value", "seasonal_naive", "holt_winters", "autoregression")


def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def _seasonal_naive(train: pd.Series, horizon: int, season_length: int = 7) -> np.ndarray:
    season = train.iloc[-season_length:].to_numpy(dtype=float)
    return np.resize(season, horizon)


def _holt_winters(train: pd.Series, horizon: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ExponentialSmoothing(
            train.astype(float),
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def _feature_row(history: list[float], date: pd.Timestamp, trend: int) -> list[float]:
    values = np.asarray(history, dtype=float)
    day = date.dayofyear
    dow = date.dayofweek
    return [
        values[-1],
        values[-7],
        values[-14],
        values[-28],
        values[-7:].mean(),
        values[-28:].mean(),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * day / 365.25),
        np.cos(2 * np.pi * day / 365.25),
        trend,
    ]


def _autoregression(train: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    values = train.to_numpy(dtype=float)
    features: list[list[float]] = []
    targets: list[float] = []
    for position in range(28, len(train)):
        features.append(_feature_row(values[:position].tolist(), train.index[position], position))
        targets.append(values[position])

    model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    model.fit(np.asarray(features), np.asarray(targets))
    history = values.tolist()
    predictions: list[float] = []
    for offset, date in enumerate(future_index):
        row = _feature_row(history, date, len(train) + offset)
        prediction = max(0.0, float(model.predict(np.asarray([row]))[0]))
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions)


def forecast(model_name: str, train: pd.Series, horizon: int) -> pd.Series:
    """Fit a named model using only *train* and forecast the next horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future_index = _future_index(train, horizon)
    if model_name == "last_value":
        values = np.repeat(float(train.iloc[-1]), horizon)
    elif model_name == "seasonal_naive":
        values = _seasonal_naive(train, horizon)
    elif model_name == "holt_winters":
        values = _holt_winters(train, horizon)
    elif model_name == "autoregression":
        values = _autoregression(train, future_index)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return pd.Series(np.clip(values, 0, None), index=future_index, name="forecast")


def rolling_origin_evaluate(
    series: pd.Series,
    initial: int = 365,
    horizon: int = 28,
    step: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on expanding windows without using future data."""
    metric_rows: list[dict[str, float | str | int]] = []
    prediction_rows: list[dict[str, float | str | pd.Timestamp | int]] = []
    fold = 0
    for origin in range(initial, len(series) - horizon + 1, step):
        train = series.iloc[:origin]
        actual = series.iloc[origin : origin + horizon]
        for model_name in MODEL_NAMES:
            predicted = forecast(model_name, train, horizon).set_axis(actual.index)
            metrics = forecast_metrics(actual, predicted)
            metric_rows.append({"fold": fold, "model": model_name, **metrics})
            for date, actual_value, predicted_value in zip(actual.index, actual, predicted):
                prediction_rows.append(
                    {
                        "fold": fold,
                        "date": date,
                        "model": model_name,
                        "actual": float(actual_value),
                        "forecast": float(predicted_value),
                        "residual": float(actual_value - predicted_value),
                    }
                )
        fold += 1
    if not metric_rows:
        raise ValueError("Series is too short for the requested validation setup.")
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)

