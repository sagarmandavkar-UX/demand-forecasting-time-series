"""Run the end-to-end analysis and print the resulting summary."""

import json
from pathlib import Path

from demand_forecasting.pipeline import run_analysis


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    summary = run_analysis(root / "data" / "raw" / "day.csv", root / "reports")
    print(json.dumps(summary, indent=2))

