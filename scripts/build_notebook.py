"""Create and execute the presentation notebook from verified pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "reports" / "summary.json").read_text(encoding="utf-8"))
    selected = summary["selected_model"].replace("_", " ").title()
    metrics = summary["holdout_metrics"]

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"""# Demand Forecasting: From Time Series to Decisions

## TL;DR

- **{selected}** won nine expanding-window validation folds.
- On the untouched 90-day holdout, MAE was **{metrics['mae']:,.0f} rentals/day** and WAPE was **{metrics['wape_pct']:.1f}%**.
- It reduced holdout MAE by **{summary['holdout_mae_improvement_vs_seasonal_naive_pct']:.1f}%** versus a weekly seasonal-naive baseline.
- Under the stated asymmetric-cost scenario, it reduced capacity cost by **{summary['scenario_cost_reduction_vs_seasonal_naive_pct']:.1f}%** versus seasonal naive.
- The nominal 80% empirical interval covered **{metrics['interval_coverage_pct']:.1f}%** of holdout observations, signaling distribution shift and a need for conservative planning.

**Decision:** use the point forecast for the base plan and the upper interval for risk-protected staffing. Treat the interval's undercoverage as an operational warning, not something to hide."""
        ),
        nbformat.v4.new_markdown_cell(
            """## Context & Methods

The objective is to forecast daily rental demand early enough to plan capacity and operations staffing. The analysis compares a last-value baseline, weekly seasonal naive, Holt-Winters, and an autoregression with lag/rolling/calendar features.

Model selection uses expanding-window validation so every forecast is trained only on dates that precede its test window. The last 90 days are held out until model selection is finished. Weather variables are excluded because realized future weather would not be known with certainty at planning time.

Uncertainty is represented by empirical residual quantiles. Business impact uses a transparent scenario: underforecasting costs 3.5 units per unplanned rental, overforecasting costs 1.0 unit per unused capacity unit, and one operations shift supports 1,200 rentals. These are illustrative assumptions, not observed company economics."""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

from demand_forecasting.data import load_daily_demand
from demand_forecasting.pipeline import run_analysis

ROOT = Path.cwd()
summary = run_analysis(ROOT / "data" / "raw" / "day.csv", ROOT / "reports")
summary"""
        ),
        nbformat.v4.new_markdown_cell("## Data\n\nThe UCI dataset contains 731 daily observations from 2011-01-01 through 2012-12-31. Validation below confirms continuous daily dates, nonnegative demand, and an internally consistent total."),
        nbformat.v4.new_code_cell(
            """data = load_daily_demand(ROOT / "data" / "raw" / "day.csv")
data[["demand", "casual", "registered"]].describe().round(1)"""
        ),
        nbformat.v4.new_code_cell("display(Image(filename=ROOT / \"reports/figures/demand_overview.png\"))"),
        nbformat.v4.new_markdown_cell("The 30-day average rises substantially from 2011 to 2012. That growth makes a static seasonal baseline risky: a strong model must represent both recent level and recurring patterns."),
        nbformat.v4.new_code_cell("display(Image(filename=ROOT / \"reports/figures/seasonality.png\"))"),
        nbformat.v4.new_markdown_cell("## Results\n\nMAE is the primary selection metric because it is in the same units as daily rentals. RMSE highlights large misses, WAPE enables relative interpretation, and bias shows systematic over- or underforecasting."),
        nbformat.v4.new_code_cell(
            """comparison = pd.read_csv(ROOT / "reports/model_comparison.csv")
comparison.assign(
    mae=comparison["mae"].round(1),
    rmse=comparison["rmse"].round(1),
    wape_pct=comparison["wape_pct"].round(1),
    bias=comparison["bias"].round(1),
)"""
        ),
        nbformat.v4.new_code_cell("display(Image(filename=ROOT / \"reports/figures/model_comparison.png\"))"),
        nbformat.v4.new_markdown_cell(
            f"""The autoregression leads rolling validation, narrowly ahead of Holt-Winters. The untouched holdout is materially harder: WAPE rises to {metrics['wape_pct']:.1f}% and bias becomes positive, meaning the model overforecasted during the late-year demand decline. This gap is evidence of nonstationarity and should influence how the forecast is used."""
        ),
        nbformat.v4.new_code_cell("display(Image(filename=ROOT / \"reports/figures/holdout_forecast.png\"))"),
        nbformat.v4.new_code_cell(
            """holdout_metrics = pd.read_csv(ROOT / "reports/holdout_metrics.csv")
holdout_metrics.round(1)"""
        ),
        nbformat.v4.new_markdown_cell("## Business impact\n\nThe cost scenario makes forecast direction matter. Underforecasting is assigned the larger penalty to represent lost service, overtime, or emergency replenishment. The values are decision weights, not dollars."),
        nbformat.v4.new_code_cell("display(Image(filename=ROOT / \"reports/figures/business_impact.png\"))"),
        nbformat.v4.new_code_cell(
            """costs = pd.read_csv(ROOT / "reports/business_impact.csv")
costs.round(1)"""
        ),
        nbformat.v4.new_markdown_cell("## 30-day planning output\n\nThe base plan uses the point forecast. The protected plan rounds the empirical upper bound into 1,200-rental operations shifts. Teams with a lower tolerance for service risk should use the protected column and revisit interval calibration as new data arrives."),
        nbformat.v4.new_code_cell(
            """plan = pd.read_csv(ROOT / "reports/future_30_day_plan.csv", parse_dates=["date"])
plan.head(10)"""
        ),
        nbformat.v4.new_code_cell(
            """plan[["forecast", "capacity_buffer", "base_shifts", "protected_shifts"]].agg(["mean", "max", "sum"]).round(1)"""
        ),
        nbformat.v4.new_markdown_cell(
            f"""## Takeaways

1. **Model choice matters:** the selected model improved holdout MAE by {summary['holdout_mae_improvement_vs_seasonal_naive_pct']:.1f}% versus repeating last week's demand.
2. **Validation must follow time:** random train/test splits would leak future conditions and overstate reliability.
3. **Uncertainty changes the plan:** the 30-day point forecast averages {summary['future_daily_average']:,.0f} rentals/day, while risk protection requires up to {summary['future_peak_protected_shifts']} shifts on the highest-capacity days.
4. **The model needs monitoring:** 65.6% coverage for a nominal 80% interval shows that historical residuals did not fully capture the holdout shift.
5. **Next step:** retrain on a rolling cadence, add forecasted weather as a scenario input, recalibrate intervals, and move from aggregate demand to station-level forecasts before operational deployment."""
        ),
    ]

    output = root / "notebooks" / "demand_forecasting.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", allow_errors=False)
    executed = client.execute(cwd=str(root))
    nbformat.write(executed, output)
    print(f"Created and executed {output}")


if __name__ == "__main__":
    main()
