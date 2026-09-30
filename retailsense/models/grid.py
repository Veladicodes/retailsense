"""Model configuration grid (XGBoost, LightGBM, Random Forest, Ridge) and a single fit/predict entry point."""
from __future__ import annotations

import numpy as np
import pandas as pd

from retailsense.config import load_config


def build_configs(fast: bool = False) -> list[dict]:
    """16 model configurations x 3 target transforms (raw / log1p / ratio-to-recent-mean).

    ``fast=True`` shrinks ensembles for unit tests only.
    """
    n = (lambda full: 30) if fast else (lambda full: full)
    cfgs: list[dict] = []
    for target in ("raw", "log", "ratio"):
        for depth in (4, 6):
            cfgs.append(dict(name=f"xgb_d{depth}_{target}", family="xgboost", target=target,
                             params=dict(max_depth=depth, learning_rate=0.05, n_estimators=n(400), subsample=0.8, colsample_bytree=0.8)))
    for target in ("raw", "log", "ratio"):
        for leaves in (15, 31):
            cfgs.append(dict(name=f"lgbm_l{leaves}_{target}", family="lightgbm", target=target,
                             params=dict(num_leaves=leaves, learning_rate=0.05, n_estimators=n(400), subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20)))
    for depth in (8, 12):
        cfgs.append(dict(name=f"rf_d{depth}_ratio", family="random_forest", target="ratio",
                         params=dict(n_estimators=n(300), max_depth=depth, min_samples_leaf=3, max_features=0.5)))
    for alpha in (1.0, 10.0):
        cfgs.append(dict(name=f"ridge_a{int(alpha)}_log", family="ridge", target="log", params=dict(alpha=alpha)))
    return cfgs


def ratio_scale(df: pd.DataFrame) -> np.ndarray:
    """Recent per-SKU demand level (past information only) used to normalise the ratio target."""
    s = df["roll_mean_13"].fillna(df["exp_mean"]).fillna(df["lag_1"]).fillna(1.0)
    return np.maximum(s.to_numpy(dtype=float), 1.0)


def _make_model(cfg: dict):
    seed = load_config().seed
    fam, p = cfg["family"], dict(cfg["params"])
    if fam == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **p)
    if fam == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)
    if fam == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(random_state=seed, n_jobs=-1, **p))
    if fam == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(**p))
    raise ValueError(f"unknown family {fam}")


def fit_predict(cfg: dict, train: pd.DataFrame, target_df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Fit on ``train`` (rows with target) and predict ``target_df``. Predictions are clipped at 0."""
    y = train["sales_qty"].to_numpy(dtype=float)
    target = cfg["target"]
    if target == "log":
        y = np.log1p(y)
    elif target == "ratio":
        y = y / ratio_scale(train)
    model = _make_model(cfg)
    model.fit(train[feature_names].to_numpy(dtype=float), y)
    pred = model.predict(target_df[feature_names].to_numpy(dtype=float))
    if target == "log":
        pred = np.expm1(pred)
    elif target == "ratio":
        pred = pred * ratio_scale(target_df)
    return np.clip(pred, 0, None)
