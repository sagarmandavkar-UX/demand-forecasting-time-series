import pandas as pd

from demand_forecasting.business import asymmetric_capacity_cost, planning_table


def test_asymmetric_cost_penalizes_shortage_more() -> None:
    actual = pd.Series([100, 100])
    forecast = pd.Series([90, 110])
    result = asymmetric_capacity_cost(actual, forecast, under_cost=3.5, over_cost=1.0)
    assert result["scenario_cost"] == 45.0


def test_planning_table_adds_uncertainty_buffer() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    table = planning_table(
        pd.Series([1200, 2400], index=index),
        pd.Series([1000, 2100], index=index),
        pd.Series([1500, 2800], index=index),
    )
    assert table["base_shifts"].tolist() == [1, 2]
    assert table["protected_shifts"].tolist() == [2, 3]

