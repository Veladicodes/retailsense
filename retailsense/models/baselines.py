"""Naive forecasting baselines built only from past-derived features."""
from __future__ import annotations

import pandas as pd

BASELINES = {
    "naive_last_week": "lag_1",
    "seasonal_naive_52": "lag_52",
    "moving_avg_4": "roll_mean_4",
    "ewm_03": "ewm_03",
}


def predict_baseline(name: str, df: pd.DataFrame) -> pd.Series:
    col = BASELINES[name]
    pred = df[col]
    if name == "seasonal_naive_52":
        pred = pred.fillna(df["lag_1"])  # no year-ago observation yet -> last week
    return pred
