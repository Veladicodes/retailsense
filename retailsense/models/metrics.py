"""Forecast accuracy metrics.

MAPE is computed only over rows whose actual demand is > 0 (MAPE is undefined at 0);
WAPE (sum |error| / sum actual) is reported alongside because it is robust to small values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mape(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    m = y > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)


def wape(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom * 100) if denom > 0 else float("nan")


def rmse(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def evaluate(df: pd.DataFrame, level: str = "sku") -> dict:
    """``df`` has sku, week_start, y_true, y_pred.

    level="sku":   every SKU-week is scored (pooled).
    level="total": predictions and actuals are summed across SKUs per week, then scored.
    """
    if level == "total":
        d = df.groupby("week_start", as_index=False)[["y_true", "y_pred"]].sum()
    elif level == "sku":
        d = df
    else:
        raise ValueError(f"unknown level: {level}")
    return {
        "mape": mape(d["y_true"], d["y_pred"]),
        "wape": wape(d["y_true"], d["y_pred"]),
        "rmse": rmse(d["y_true"], d["y_pred"]),
        "n_rows": int(len(d)),
    }
