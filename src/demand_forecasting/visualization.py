"""Consistent, presentation-ready project visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLORS = {"actual": "#1f2937", "forecast": "#2563eb", "interval": "#93c5fd", "accent": "#f97316"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.titleweight": "bold"})


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_overview(data: pd.DataFrame, path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(data.index, data["demand"], color="#94a3b8", linewidth=1, alpha=0.7, label="Daily demand")
    ax.plot(data.index, data["demand"].rolling(30).mean(), color=COLORS["forecast"], linewidth=2.5, label="30-day average")
    ax.set(title="Daily bike-rental demand shows growth and seasonality", xlabel="Date", ylabel="Rentals per day")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_seasonality(data: pd.DataFrame, path: Path) -> None:
    set_style()
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday = data.groupby(data.index.dayofweek)["demand"].mean().reindex(range(7))
    monthly = data.groupby(data.index.month)["demand"].mean().reindex(range(1, 13))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=weekday_names, y=weekday.values, color=COLORS["forecast"], ax=axes[0])
    axes[0].set(title="Average demand by weekday", xlabel="Weekday", ylabel="Average daily rentals")
    axes[1].plot(range(1, 13), monthly.values, color=COLORS["accent"], marker="o", linewidth=2.5)
    axes[1].set(title="Average demand by month", xlabel="Month", ylabel="Average daily rentals", xticks=range(1, 13))
    fig.suptitle("Recurring calendar patterns inform the forecast", y=1.04, fontweight="bold")
    _save(fig, path)


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    set_style()
    ordered = metrics.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(ordered["model"], ordered["mae"], color=[COLORS["forecast"]] + ["#94a3b8"] * (len(ordered) - 1))
    ax.bar_label(bars, fmt="%.0f", padding=8)
    ax.set(title="Rolling validation identifies the most accurate model", xlabel="Mean absolute error (rentals/day) — lower is better", ylabel="Model")
    ax.set_xlim(0, ordered["mae"].max() * 1.2)
    _save(fig, path)


def plot_holdout(forecast_frame: pd.DataFrame, path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(forecast_frame.index, forecast_frame["lower_80"], forecast_frame["upper_80"], color=COLORS["interval"], alpha=0.45, label="Empirical 80% interval")
    ax.plot(forecast_frame.index, forecast_frame["actual"], color=COLORS["actual"], linewidth=2, label="Actual")
    ax.plot(forecast_frame.index, forecast_frame["forecast"], color=COLORS["forecast"], linewidth=2.2, label="Forecast")
    ax.set(title="Selected model on the untouched 90-day holdout", xlabel="Date", ylabel="Rentals per day")
    ax.legend(frameon=False, ncol=3)
    _save(fig, path)


def plot_business_cost(costs: pd.DataFrame, path: Path) -> None:
    set_style()
    ordered = costs.sort_values("scenario_cost")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(ordered["model"], ordered["scenario_cost"], color=[COLORS["forecast"]] + ["#94a3b8"] * (len(ordered) - 1))
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in ordered["scenario_cost"]], padding=8)
    ax.set(title="Forecast quality changes hypothetical capacity cost", xlabel="Scenario cost units — lower is better", ylabel="Model")
    ax.set_xlim(0, ordered["scenario_cost"].max() * 1.22)
    _save(fig, path)

