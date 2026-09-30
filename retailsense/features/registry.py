"""The 67-feature registry for one-week-ahead SKU demand forecasting.

Every feature for week ``t`` is computed from information available at the end of
week ``t-1`` or from deterministic calendar attributes of week ``t``. Nothing in
this module reads the target (``sales_qty``) or any covariate of week ``t``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR = [
    "week_of_year", "month", "quarter", "week_sin", "week_cos", "month_sin", "month_cos",
    "is_holiday_season", "weeks_to_christmas", "t_index",
]
LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
WINDOWS = [4, 8, 13, 26]
LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLLING = [f"roll_{stat}_{w}" for w in WINDOWS for stat in ("mean", "std", "min", "max")]
EWM = ["ewm_03", "ewm_05"]
MOMENTUM = [
    "diff_1", "diff_4", "pct_change_1", "ratio_4_13", "ratio_4_26", "ratio_1_8",
    "cv_8", "slope_8", "zero_share_13", "seas52_mean3",
]
PRICE = ["price_lag_1", "price_lag_4", "price_pct_change_1", "price_rel_mean13", "price_std_8", "price_max_13"]
COVARIATES = ["n_invoices_lag_1", "n_customers_lag_1", "returns_qty_lag_1", "invoices_mean_4", "qty_per_invoice_lag_1"]
EXPANDING = ["exp_mean", "exp_std", "exp_max", "sku_age_weeks"]
MARKET = ["total_sales_lag_1", "total_sales_mean4", "sku_share_mean13"]
FLAGS = ["week_of_month", "is_pre_christmas", "is_post_christmas"]

FEATURE_NAMES: list[str] = (
    CALENDAR + LAG_FEATURES + ROLLING + EWM + MOMENTUM + PRICE + COVARIATES + EXPANDING + MARKET + FLAGS
)
TARGET = "sales_qty"
KEYS = ["sku", "week_start"]


def weeks_to_christmas(week_start: pd.Series) -> pd.Series:
    """Weeks from week_start to the next Dec 25 (rolls over to next year once passed)."""
    ws = pd.to_datetime(week_start)
    xmas = pd.to_datetime(ws.dt.year.astype(str) + "-12-25")
    xmas = xmas.where(ws <= xmas, xmas + pd.DateOffset(years=1))
    return (xmas - ws).dt.days / 7.0


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.where(b != 0)
    return out.replace([np.inf, -np.inf], np.nan)


def _slope(window: np.ndarray) -> float:
    if np.isnan(window).any():
        return np.nan
    x = np.arange(len(window), dtype=float)
    x -= x.mean()
    return float((x * (window - window.mean())).sum() / (x ** 2).sum())


def _calendar(ws: pd.Series, t_index: pd.Series) -> pd.DataFrame:
    woy = ws.dt.isocalendar().week.astype(float)
    c = pd.DataFrame(index=ws.index)
    c["week_of_year"] = woy
    c["month"] = ws.dt.month.astype(float)
    c["quarter"] = ws.dt.quarter.astype(float)
    c["week_sin"] = np.sin(2 * np.pi * woy / 52)
    c["week_cos"] = np.cos(2 * np.pi * woy / 52)
    c["month_sin"] = np.sin(2 * np.pi * ws.dt.month / 12)
    c["month_cos"] = np.cos(2 * np.pi * ws.dt.month / 12)
    c["is_holiday_season"] = ws.dt.month.isin([11, 12]).astype(float)
    c["weeks_to_christmas"] = weeks_to_christmas(ws)
    c["t_index"] = t_index.astype(float)
    return c


def _sku_frame(g: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week_start").reset_index(drop=True)
    y = g[TARGET].astype(float)
    ws = g["week_start"]
    f = _calendar(ws, pd.Series(np.arange(len(g)), index=g.index))

    for k in LAGS:
        f[f"lag_{k}"] = y.shift(k)

    y1 = y.shift(1)
    for w in WINDOWS:
        r = y1.rolling(w, min_periods=2)
        f[f"roll_mean_{w}"] = r.mean()
        f[f"roll_std_{w}"] = r.std()
        f[f"roll_min_{w}"] = r.min()
        f[f"roll_max_{w}"] = r.max()
    f["ewm_03"] = y1.ewm(alpha=0.3, adjust=False).mean()
    f["ewm_05"] = y1.ewm(alpha=0.5, adjust=False).mean()

    f["diff_1"] = f["lag_1"] - f["lag_2"]
    f["diff_4"] = f["lag_1"] - y.shift(5)
    f["pct_change_1"] = _safe_div(f["lag_1"] - f["lag_2"], f["lag_2"])
    f["ratio_4_13"] = _safe_div(f["roll_mean_4"], f["roll_mean_13"])
    f["ratio_4_26"] = _safe_div(f["roll_mean_4"], f["roll_mean_26"])
    f["ratio_1_8"] = _safe_div(f["lag_1"], f["roll_mean_8"])
    f["cv_8"] = _safe_div(f["roll_std_8"], f["roll_mean_8"])
    f["slope_8"] = y1.rolling(8, min_periods=8).apply(_slope, raw=True)
    f["zero_share_13"] = (y1 == 0).astype(float).where(y1.notna()).rolling(13, min_periods=2).mean()
    f["seas52_mean3"] = pd.concat([y.shift(51), y.shift(52), y.shift(53)], axis=1).mean(axis=1, skipna=False)

    price = g["avg_price"].astype(float).ffill()
    p1 = price.shift(1)
    f["price_lag_1"] = p1
    f["price_lag_4"] = price.shift(4)
    f["price_pct_change_1"] = _safe_div(p1 - price.shift(2), price.shift(2))
    f["price_rel_mean13"] = _safe_div(p1, p1.rolling(13, min_periods=2).mean())
    f["price_std_8"] = p1.rolling(8, min_periods=2).std()
    f["price_max_13"] = p1.rolling(13, min_periods=2).max()

    inv1 = g["n_invoices"].astype(float).shift(1)
    f["n_invoices_lag_1"] = inv1
    f["n_customers_lag_1"] = g["n_customers"].astype(float).shift(1)
    f["returns_qty_lag_1"] = g["returns_qty"].astype(float).shift(1)
    f["invoices_mean_4"] = inv1.rolling(4, min_periods=1).mean()
    f["qty_per_invoice_lag_1"] = _safe_div(f["lag_1"], inv1)

    f["exp_mean"] = y1.expanding(min_periods=2).mean()
    f["exp_std"] = y1.expanding(min_periods=2).std()
    f["exp_max"] = y1.expanding(min_periods=1).max()
    sold_before = (y > 0).cummax().shift(1, fill_value=False)
    f["sku_age_weeks"] = sold_before.astype(float).cumsum()

    m = market.reindex(ws.values)
    f["total_sales_lag_1"] = m["total_lag_1"].to_numpy()
    f["total_sales_mean4"] = m["total_mean4"].to_numpy()
    f["sku_share_mean13"] = _safe_div(f["roll_mean_13"], pd.Series(m["total_mean13"].to_numpy(), index=f.index))

    f["week_of_month"] = ((ws.dt.day - 1) // 7 + 1).astype(float)
    wtc = f["weeks_to_christmas"]
    f["is_pre_christmas"] = ((wtc >= 1) & (wtc <= 4)).astype(float)
    f["is_post_christmas"] = (((ws.dt.month == 12) & (ws.dt.day >= 26)) | ((ws.dt.month == 1) & (ws.dt.day <= 10))).astype(float)

    out = g[KEYS + [TARGET]].copy()
    return pd.concat([out, f[FEATURE_NAMES]], axis=1)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Dense SKU x week panel -> (sku, week_start, sales_qty, 67 features)."""
    panel = panel.copy()
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    total = panel.groupby("week_start")[TARGET].sum().sort_index().astype(float)
    t1 = total.shift(1)
    market = pd.DataFrame({
        "total_lag_1": t1,
        "total_mean4": t1.rolling(4, min_periods=1).mean(),
        "total_mean13": t1.rolling(13, min_periods=2).mean(),
    })
    frames = [_sku_frame(g, market) for _, g in panel.groupby("sku", sort=True)]
    out = pd.concat(frames, ignore_index=True)
    out[FEATURE_NAMES] = out[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    return out


def assign_split(features: pd.DataFrame, b) -> pd.DataFrame:
    """Label rows train / val / test using SQL-derived boundaries."""
    out = features.copy()
    ws = pd.to_datetime(out["week_start"])
    out["split"] = np.select(
        [ws >= b.test_start, ws >= b.val_start, ws <= b.train_end],
        ["test", "val", "train"],
        default="gap",
    )
    return out[out["split"] != "gap"].reset_index(drop=True)
