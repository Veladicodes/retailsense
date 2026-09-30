"""Model configuration grid (XGBoost, LightGBM, Random Forest, Ridge) and a single fit/predict entry point."""
from __future__ import annotations

import numpy as np
import pandas as pd

from retailsense.config import load_config


def build_configs(fast: bool = False) -> list[dict]:
    """16 model configurations. ``fast=True`` shrinks ensembles for unit tests only."""
    n = (lambda full: 30) if fast else (lambda full: full)
    cfgs: list[dict] = []
    for depth, lr, est in [(4, 0.05, 400), (6, 0.05, 400), (4, 0.10, 200), (6, 0.10, 200)]:
        cfgs.append(dict(name=f"xgb_d{depth}_lr{int(lr * 100):02d}_n{est}", family="xgboost", log_target=False,
                         params=dict(max_depth=depth, learning_rate=lr, n_estimators=n(est), subsample=0.8, colsample_bytree=0.8)))
    for depth in (4, 6):
        cfgs.append(dict(name=f"xgb_d{depth}_lr05_n400_log", family="xgboost", log_target=True,
                         params=dict(max_depth=depth, learning_rate=0.05, n_estimators=n(400), subsample=0.8, colsample_bytree=0.8)))
    for leaves in (15, 31, 63):
        cfgs.append(dict(name=f"lgbm_l{leaves}", family="lightgbm", log_target=False,
                         params=dict(num_leaves=leaves, learning_rate=0.05, n_estimators=n(400), subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20)))
    for leaves in (15, 31):
        cfgs.append(dict(name=f"lgbm_l{leaves}_log", family="lightgbm", log_target=True,
                         params=dict(num_leaves=leaves, learning_rate=0.05, n_estimators=n(400), subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20)))
    for depth in (8, 12):
        cfgs.append(dict(name=f"rf_d{depth}", family="random_forest", log_target=False,
                         params=dict(n_estimators=n(300), max_depth=depth, min_samples_leaf=3, max_features=0.5)))
    for alpha in (1.0, 10.0):
        cfgs.append(dict(name=f"ridge_a{int(alpha)}", family="ridge", log_target=True, params=dict(alpha=alpha)))
    return cfgs


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
    if cfg["log_target"]:
        y = np.log1p(y)
    model = _make_model(cfg)
    model.fit(train[feature_names].to_numpy(dtype=float), y)
    pred = model.predict(target_df[feature_names].to_numpy(dtype=float))
    if cfg["log_target"]:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)
