"""End-to-end forecasting analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .business import asymmetric_capacity_cost, planning_table
from .data import load_daily_demand
from .evaluation import forecast_metrics, interval_coverage
from .models import MODEL_NAMES, forecast, rolling_origin_evaluate
from .visualization import plot_business_cost, plot_holdout, plot_model_comparison, plot_overview, plot_seasonality


def _round_dict(values: dict[str, float], digits: int = 2) -> dict[str, float]:
    return {key: round(float(value), digits) for key, value in values.items()}


def run_analysis(data_path: Path, reports_dir: Path, holdout_days: int = 90) -> dict[str, object]:
    """Run data checks, model selection, holdout evaluation, and future planning."""
    data_path = Path(data_path)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_daily_demand(data_path)
    series = data["demand"].astype(float)
    development = series.iloc[:-holdout_days]
    holdout = series.iloc[-holdout_days:]

    fold_metrics, cv_predictions = rolling_origin_evaluate(development, initial=365, horizon=28, step=28)
    comparison = (
        fold_metrics.groupby("model", as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), wape_pct=("wape_pct", "mean"), bias=("bias", "mean"), folds=("fold", "nunique"))
        .sort_values("mae")
        .reset_index(drop=True)
    )
    selected_model = str(comparison.iloc[0]["model"])
    calibration_residuals = cv_predictions.loc[cv_predictions["model"] == selected_model, "residual"]
    lower_offset, upper_offset = calibration_residuals.quantile([0.10, 0.90]).to_numpy()

    holdout_rows: list[dict[str, float | str]] = []
    holdout_forecasts: dict[str, pd.Series] = {}
    for model_name in MODEL_NAMES:
        model_forecast = forecast(model_name, development, holdout_days).set_axis(holdout.index)
        holdout_forecasts[model_name] = model_forecast
        holdout_rows.append({"model": model_name, **forecast_metrics(holdout, model_forecast)})
    holdout_metrics = pd.DataFrame(holdout_rows).sort_values("mae").reset_index(drop=True)

    selected_forecast = holdout_forecasts[selected_model]
    holdout_frame = pd.DataFrame({"actual": holdout, "forecast": selected_forecast})
    holdout_frame["lower_80"] = np.clip(selected_forecast + lower_offset, 0, None)
    holdout_frame["upper_80"] = np.clip(selected_forecast + upper_offset, 0, None)
    selected_metrics = forecast_metrics(holdout, selected_forecast)
    selected_metrics["interval_coverage_pct"] = interval_coverage(holdout, holdout_frame["lower_80"], holdout_frame["upper_80"])

    cost_rows: list[dict[str, float | str]] = []
    for model_name, model_forecast in holdout_forecasts.items():
        cost_rows.append({"model": model_name, **asymmetric_capacity_cost(holdout, model_forecast)})
    business_costs = pd.DataFrame(cost_rows).sort_values("scenario_cost").reset_index(drop=True)

    future = forecast(selected_model, series, 30)
    holdout_residuals = holdout - selected_forecast
    future_lower_offset, future_upper_offset = holdout_residuals.quantile([0.10, 0.90]).to_numpy()
    future_lower = (future + future_lower_offset).clip(lower=0)
    future_upper = (future + future_upper_offset).clip(lower=0)
    plan = planning_table(future, future_lower, future_upper)

    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    fold_metrics.to_csv(reports_dir / "cross_validation_metrics.csv", index=False)
    holdout_metrics.to_csv(reports_dir / "holdout_metrics.csv", index=False)
    holdout_frame.to_csv(reports_dir / "holdout_forecast.csv", index_label="date")
    business_costs.to_csv(reports_dir / "business_impact.csv", index=False)
    plan.to_csv(reports_dir / "future_30_day_plan.csv")

    plot_overview(data, figures_dir / "demand_overview.png")
    plot_seasonality(data, figures_dir / "seasonality.png")
    plot_model_comparison(comparison, figures_dir / "model_comparison.png")
    plot_holdout(holdout_frame, figures_dir / "holdout_forecast.png")
    plot_business_cost(business_costs, figures_dir / "business_impact.png")

    seasonal_mae = float(holdout_metrics.loc[holdout_metrics["model"] == "seasonal_naive", "mae"].iloc[0])
    best_cost = float(business_costs.loc[business_costs["model"] == selected_model, "scenario_cost"].iloc[0])
    seasonal_cost = float(business_costs.loc[business_costs["model"] == "seasonal_naive", "scenario_cost"].iloc[0])
    summary: dict[str, object] = {
        "observations": int(len(data)),
        "date_start": data.index.min().date().isoformat(),
        "date_end": data.index.max().date().isoformat(),
        "selected_model": selected_model,
        "cv_metrics": _round_dict(comparison.iloc[0][["mae", "rmse", "wape_pct", "bias"]].to_dict()),
        "holdout_metrics": _round_dict(selected_metrics),
        "holdout_mae_improvement_vs_seasonal_naive_pct": round(100 * (seasonal_mae - selected_metrics["mae"]) / seasonal_mae, 2),
        "scenario_cost_reduction_vs_seasonal_naive_pct": round(100 * (seasonal_cost - best_cost) / seasonal_cost, 2),
        "future_30_day_total": int(plan["forecast"].sum()),
        "future_daily_average": round(float(plan["forecast"].mean()), 1),
        "future_peak_forecast": int(plan["forecast"].max()),
        "future_peak_protected_shifts": int(plan["protected_shifts"].max()),
        "assumptions": {
            "underforecast_unit_cost": 3.5,
            "overforecast_unit_cost": 1.0,
            "rides_per_operations_shift": 1200,
            "prediction_interval": "Empirical 80% interval from historical out-of-sample residuals",
        },
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

