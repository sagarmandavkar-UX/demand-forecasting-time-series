"""Download and validate the UCI daily bike-sharing demand data."""

from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download_daily_data(destination: Path) -> Path:
    """Download day.csv from UCI and store it at *destination*."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_DATASET_URL, timeout=30, context=context) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        member = next(name for name in zipped.namelist() if name.endswith("day.csv"))
        destination.write_bytes(zipped.read(member))
    return destination


def load_daily_demand(path: Path) -> pd.DataFrame:
    """Load daily demand and enforce the data assumptions used by the models."""
    path = Path(path)
    if not path.exists():
        download_daily_data(path)

    frame = pd.read_csv(path, parse_dates=["dteday"])
    required = {"dteday", "cnt", "casual", "registered"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.rename(columns={"dteday": "date", "cnt": "demand"})
    frame = frame.sort_values("date").set_index("date")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    if not frame.index.equals(expected):
        raise ValueError("Dates must be unique, ordered, and daily without gaps.")
    if (frame["demand"] < 0).any():
        raise ValueError("Demand cannot be negative.")
    if not (frame["casual"] + frame["registered"]).equals(frame["demand"]):
        raise ValueError("Total demand must equal casual plus registered demand.")
    return frame
