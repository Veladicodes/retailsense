"""Pipeline tasks shared by the CLI, Airflow DAG and tests.

ingest -> validate -> build_features -> train -> forecast -> publish (S3)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from retailsense.config import Config
from retailsense.data.db import get_engine, init_schema
from retailsense.data.ingest import build_calendar, clean_transactions, load_raw_sales
from retailsense.data.queries import split_boundaries, weekly_panel
from retailsense.features.registry import FEATURE_NAMES, assign_split, build_features
from retailsense.models.experiment import log_runs, run_experiments
from retailsense.models.grid import build_configs, fit_predict
from retailsense.reporting import write_report

log = logging.getLogger(__name__)


class DataValidationError(RuntimeError):
    pass


def artifacts_dir(cfg: Config) -> Path:
    p = Path(cfg.data_dir) / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_raw(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    return pd.read_csv(path)


def ingest(cfg: Config, raw_path: str | Path) -> dict:
    engine = get_engine(cfg.db_url)
    init_schema(engine)
    rows = load_raw_sales(engine, clean_transactions(_read_raw(Path(raw_path))))
    weeks = build_calendar(engine)
    log.info("ingested %d rows, %d complete weeks", rows, len(weeks))
    return {"rows": rows, "weeks": len(weeks)}


def validate(cfg: Config) -> dict:
    engine = get_engine(cfg.db_url)
    with engine.connect() as c:
        rows = c.execute(text("SELECT COUNT(*) FROM raw_sales")).scalar()
        weeks = c.execute(text("SELECT COUNT(*) FROM calendar_weeks")).scalar()
    if not rows:
        raise DataValidationError("raw_sales is empty - run ingest first")
    need = cfg.holdout_weeks + cfg.validation_weeks + 26
    if weeks < need:
        raise DataValidationError(f"only {weeks} complete weeks; need at least {need}")
    panel = weekly_panel(engine, top_n=cfg.top_n_skus, min_weeks=cfg.min_weeks)
    if panel.empty:
        raise DataValidationError("no SKU has enough weekly history")
    dup = int(panel.duplicated(["sku", "week_start"]).sum())
    if dup:
        raise DataValidationError(f"{dup} duplicate sku-week rows")
    return {"rows": int(rows), "weeks": int(weeks), "skus": int(panel["sku"].nunique()), "duplicate_sku_weeks": dup}


def build_features_task(cfg: Config) -> dict:
    engine = get_engine(cfg.db_url)
    panel = weekly_panel(engine, top_n=cfg.top_n_skus, min_weeks=cfg.min_weeks)
    feats = assign_split(build_features(panel), split_boundaries(engine, cfg.holdout_weeks, cfg.validation_weeks))
    out = feats.copy()
    out["week_start"] = pd.to_datetime(out["week_start"]).dt.strftime("%Y-%m-%d")
    out.to_sql("features", engine, if_exists="replace", index=False, chunksize=2000)
    return {"rows": len(out), "n_features": len(FEATURE_NAMES), "skus": int(out["sku"].nunique())}


def _load_features(cfg: Config) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM features", get_engine(cfg.db_url))
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def train(cfg: Config, fast: bool = False) -> dict:
    engine = get_engine(cfg.db_url)
    result = run_experiments(_load_features(cfg), fast=fast)
    run_id = log_runs(engine, result)
    art = artifacts_dir(cfg)
    write_report(result, art / "experiments.md", art / "summary.json")
    (art / "run.json").write_text(json.dumps({"run_id": run_id, "best_model": result.summary["best_model"], "fast": fast}), encoding="utf-8")
    best = result.predictions[(result.predictions.model == result.summary["best_model"]) & (result.predictions.split == "test")]
    best.to_csv(art / "test_predictions.csv", index=False)
    return {"run_id": run_id, "summary": result.summary}


def forecast_next_week(cfg: Config, fast: bool = False) -> dict:
    """Fit the validation-selected model on all history and predict the week after the last complete week."""
    engine = get_engine(cfg.db_url)
    art = artifacts_dir(cfg)
    run = json.loads((art / "run.json").read_text(encoding="utf-8"))
    config = next(c for c in build_configs(fast=fast) if c["name"] == run["best_model"])

    panel = weekly_panel(engine, top_n=cfg.top_n_skus, min_weeks=cfg.min_weeks)
    next_week = panel["week_start"].max() + pd.Timedelta(days=7)
    future = pd.DataFrame({"sku": panel["sku"].unique()})
    future["week_start"] = next_week
    future["sales_qty"] = 0.0  # placeholder: never read by the features of its own row
    future[["returns_qty", "n_invoices", "n_customers"]] = 0
    future["avg_price"] = float("nan")
    feats = build_features(pd.concat([panel, future], ignore_index=True))

    hist = feats[(feats.week_start < next_week) & feats["lag_1"].notna()]
    target = feats[feats.week_start == next_week].copy()
    pred = fit_predict(config, hist, target, FEATURE_NAMES)

    out = pd.DataFrame({
        "run_id": run["run_id"], "sku": target["sku"].to_numpy(),
        "week_start": next_week.strftime("%Y-%m-%d"), "split": "future",
        "y_true": float("nan"), "y_pred": pred,
    })
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM forecasts WHERE split = 'future'"))
    out.to_sql("forecasts", engine, if_exists="append", index=False)
    out[["sku", "week_start", "y_pred"]].to_csv(art / "next_week_forecast.csv", index=False)
    return {"n_skus": int(len(out)), "week_start": next_week.strftime("%Y-%m-%d"), "model": run["best_model"]}


def publish(cfg: Config, store, run_date: str | None = None) -> list[str]:
    run_date = run_date or date.today().isoformat()
    store.ensure_bucket()
    keys = []
    for f in sorted(artifacts_dir(cfg).iterdir()):
        if f.is_file():
            key = f"runs/{run_date}/{f.name}"
            store.put_file(f, key)
            keys.append(key)
    return keys


def run_all(cfg: Config, raw_path, store=None, fast: bool = False, run_date: str | None = None) -> dict:
    ingest(cfg, raw_path)
    validate(cfg)
    build_features_task(cfg)
    trained = train(cfg, fast=fast)
    forecast_next_week(cfg, fast=fast)
    published = publish(cfg, store, run_date) if store is not None else []
    return {"summary": trained["summary"], "run_id": trained["run_id"], "published": published}
