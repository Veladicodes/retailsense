"""Experiment protocol: baselines + 16 configs + top-3 ensemble, selection on validation only, final score on holdout.

1. Every config is fit on ``train`` and scored on ``val``  -> used ONLY to choose the model.
2. Every config is refit on ``train + val`` and scored on the untouched ``test`` holdout.
3. The headline numbers are the test scores of (a) the best baseline chosen on val and
   (b) the best model chosen on val. Test scores of all other models are reported for transparency.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from retailsense.features.registry import FEATURE_NAMES
from retailsense.models.baselines import BASELINES, predict_baseline
from retailsense.models.grid import build_configs, fit_predict
from retailsense.models.metrics import evaluate


@dataclass
class ExperimentResult:
    runs: pd.DataFrame          # model, family, split, level, mape, wape, rmse, n_rows, params_json
    predictions: pd.DataFrame   # model, sku, week_start, split, y_true, y_pred
    summary: dict


def _pred_frame(df: pd.DataFrame, pred, model: str, split: str) -> pd.DataFrame:
    return pd.DataFrame({
        "model": model, "sku": df["sku"].to_numpy(), "week_start": df["week_start"].to_numpy(),
        "split": split, "y_true": df["sales_qty"].to_numpy(dtype=float), "y_pred": np.asarray(pred, dtype=float),
    })


def _score(pf: pd.DataFrame, family: str, params_json: str) -> list[dict]:
    rows = []
    for level in ("sku", "total"):
        m = evaluate(pf, level=level)
        rows.append(dict(model=pf["model"].iloc[0], family=family, split=pf["split"].iloc[0], level=level,
                         params_json=params_json, **m))
    return rows


def run_experiments(features: pd.DataFrame, fast: bool = False) -> ExperimentResult:
    feats = features[features["lag_1"].notna()].copy()
    train = feats[feats.split == "train"]
    val = feats[feats.split == "val"]
    test = feats[feats.split == "test"]
    trainval = pd.concat([train, val])
    configs = build_configs(fast=fast)

    runs: list[dict] = []
    preds: list[pd.DataFrame] = []

    def register(frame: pd.DataFrame, family: str, params: dict):
        preds.append(frame)
        runs.extend(_score(frame, family, json.dumps(params, sort_keys=True)))

    # baselines need no fitting: score them on both splits
    for name in BASELINES:
        for split_name, part in (("val", val), ("test", test)):
            register(_pred_frame(part, predict_baseline(name, part), name, split_name), "baseline", {"feature": BASELINES[name]})

    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    for cfg in configs:
        pv = fit_predict(cfg, train, val, FEATURE_NAMES)
        pt = fit_predict(cfg, trainval, test, FEATURE_NAMES)
        val_preds[cfg["name"]], test_preds[cfg["name"]] = pv, pt
        register(_pred_frame(val, pv, cfg["name"], "val"), cfg["family"], cfg["params"])
        register(_pred_frame(test, pt, cfg["name"], "test"), cfg["family"], cfg["params"])

    # Top-3 ensemble: members are ranked on VALIDATION SKU-level WAPE only.
    val_wape = {r["model"]: r["wape"] for r in runs if r["split"] == "val" and r["level"] == "sku" and r["family"] != "baseline"}
    members = sorted(val_wape, key=val_wape.get)[:3]
    ens_params = {"members": members}
    register(_pred_frame(val, np.mean([val_preds[m] for m in members], axis=0), "ensemble_top3", "val"), "ensemble", ens_params)
    register(_pred_frame(test, np.mean([test_preds[m] for m in members], axis=0), "ensemble_top3", "test"), "ensemble", ens_params)

    runs_df = pd.DataFrame(runs)
    val_sku = runs_df[(runs_df.split == "val") & (runs_df.level == "sku")]
    best_model = val_sku[val_sku.family != "baseline"].set_index("model")["wape"].idxmin()
    baseline_model = val_sku[val_sku.family == "baseline"].set_index("model")["wape"].idxmin()

    def test_metric(model: str, level: str, key: str = "mape") -> float:
        r = runs_df[(runs_df.model == model) & (runs_df.split == "test") & (runs_df.level == level)]
        return float(r[key].iloc[0])

    summary = {
        "baseline_model": baseline_model,
        "best_model": best_model,
        "baseline_mape_sku": test_metric(baseline_model, "sku"),
        "best_mape_sku": test_metric(best_model, "sku"),
        "baseline_mape_total": test_metric(baseline_model, "total"),
        "best_mape_total": test_metric(best_model, "total"),
        "baseline_wape_sku": test_metric(baseline_model, "sku", "wape"),
        "best_wape_sku": test_metric(best_model, "sku", "wape"),
        "worst_baseline_mape_sku": float(max(test_metric(b, "sku") for b in BASELINES)),
        "n_configs": len(configs) + 1,  # fitted configs + the top-3 ensemble
        "n_baselines": len(BASELINES),
        "n_features": len(FEATURE_NAMES),
        "n_skus": int(feats["sku"].nunique()),
        "n_test_weeks": int(test["week_start"].nunique()),
        "n_val_weeks": int(val["week_start"].nunique()),
        "test_start": str(pd.Timestamp(test["week_start"].min()).date()),
        "test_end": str(pd.Timestamp(test["week_start"].max()).date()),
        "selection_rule": "lowest validation SKU-level WAPE (models and baselines alike); holdout scored once after refit on train+val",
    }
    return ExperimentResult(runs=runs_df, predictions=pd.concat(preds, ignore_index=True), summary=summary)


def log_runs(engine: Engine, result: ExperimentResult, run_id: str | None = None) -> str:
    """Persist metrics (model_runs) and best-model forecasts (forecasts) to SQL."""
    run_id = run_id or uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mr = result.runs.copy()
    mr["params_json"] = mr["params_json"].astype(str)
    mr["run_id"] = run_id
    mr["model_name"] = mr["model"]
    mr["created_at"] = now
    mr[["run_id", "model_name", "family", "params_json", "split", "level", "mape", "wape", "rmse", "n_rows", "created_at"]] \
        .to_sql("model_runs", engine, if_exists="append", index=False)
    keep = {result.summary["best_model"], result.summary["baseline_model"]}
    fc = result.predictions[result.predictions.model.isin(keep)].copy()
    fc["run_id"] = run_id
    fc["week_start"] = pd.to_datetime(fc["week_start"]).dt.strftime("%Y-%m-%d")
    fc[["run_id", "sku", "week_start", "split", "y_true", "y_pred"]].to_sql("forecasts", engine, if_exists="append", index=False)
    return run_id
