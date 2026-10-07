"""Download the project dataset from UCI."""

from pathlib import Path

from demand_forecasting.data import download_daily_data


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    path = download_daily_data(root / "data" / "raw" / "day.csv")
    print(f"Downloaded {path}")

