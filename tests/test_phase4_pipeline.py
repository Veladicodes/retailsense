import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from tests.conftest import make_transactions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("RETAILSENSE_S3_ENDPOINT", raising=False)


@pytest.fixture
def cfg(tmp_path):
    from retailsense.config import load_config

    return dataclasses.replace(
        load_config(), data_dir=tmp_path, db_url=f"sqlite:///{(tmp_path / 'rs.db').as_posix()}",
        top_n_skus=6, holdout_weeks=8, validation_weeks=8, s3_bucket="rs-test", s3_endpoint=None,
    )


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "raw.csv"
    make_transactions(n_weeks=110, skus=tuple(f"S{i}" for i in range(6))).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ S3 store
def test_s3_store_roundtrip(aws_env, tmp_path):
    from moto import mock_aws
    from retailsense.storage.s3 import S3Store

    with mock_aws():
        store = S3Store("b1", endpoint=None)
        store.ensure_bucket()
        src = tmp_path / "a.txt"
        src.write_text("hello")
        store.put_file(src, "x/a.txt")
        assert store.exists("x/a.txt") and not store.exists("x/nope.txt")
        assert store.list("x/") == ["x/a.txt"]
        dst = tmp_path / "out.txt"
        store.get_file("x/a.txt", dst)
        assert dst.read_text() == "hello"


def test_s3_ensure_bucket_is_idempotent(aws_env):
    from moto import mock_aws
    from retailsense.storage.s3 import S3Store

    with mock_aws():
        s = S3Store("b2")
        s.ensure_bucket()
        s.ensure_bucket()


# ------------------------------------------------------------------ pipeline tasks
def test_ingest_loads_sql_and_calendar(cfg, raw_csv):
    from retailsense.pipeline import tasks

    out = tasks.ingest(cfg, raw_csv)
    assert out["rows"] > 0 and out["weeks"] > 50
    eng = create_engine(cfg.db_url)
    with eng.connect() as c:
        assert c.execute(text("select count(*) from raw_sales")).scalar() == out["rows"]


def test_validate_passes_on_good_data_and_fails_on_empty(cfg, raw_csv):
    from retailsense.pipeline import tasks

    tasks.ingest(cfg, raw_csv)
    ok = tasks.validate(cfg)
    assert ok["skus"] >= 6 and ok["duplicate_sku_weeks"] == 0

    empty = dataclasses.replace(cfg, db_url=f"sqlite:///{(cfg.data_dir / 'empty.db').as_posix()}")
    from retailsense.data.db import get_engine, init_schema

    init_schema(get_engine(empty.db_url))
    with pytest.raises(tasks.DataValidationError):
        tasks.validate(empty)


def test_build_features_writes_features_table(cfg, raw_csv):
    from retailsense.features.registry import FEATURE_NAMES
    from retailsense.pipeline import tasks

    tasks.ingest(cfg, raw_csv)
    out = tasks.build_features_task(cfg)
    assert out["n_features"] == 67
    cols = set(pd.read_sql("select * from features limit 1", create_engine(cfg.db_url)).columns)
    assert set(FEATURE_NAMES) <= cols and {"sku", "week_start", "sales_qty", "split"} <= cols


def test_train_logs_runs_and_writes_artifacts(cfg, raw_csv):
    from retailsense.pipeline import tasks

    tasks.ingest(cfg, raw_csv)
    tasks.build_features_task(cfg)
    out = tasks.train(cfg, fast=True)
    art = cfg.data_dir / "artifacts"
    assert (art / "experiments.md").exists() and (art / "summary.json").exists()
    with create_engine(cfg.db_url).connect() as c:
        assert c.execute(text("select count(*) from model_runs where run_id=:r"), {"r": out["run_id"]}).scalar() > 0
    assert out["summary"]["n_configs"] >= 15


def test_forecast_next_week_one_row_per_sku(cfg, raw_csv):
    from retailsense.pipeline import tasks

    tasks.ingest(cfg, raw_csv)
    tasks.build_features_task(cfg)
    tasks.train(cfg, fast=True)
    out = tasks.forecast_next_week(cfg, fast=True)
    eng = create_engine(cfg.db_url)
    df = pd.read_sql("select * from forecasts where split='future'", eng)
    assert len(df) == out["n_skus"] == 6
    last_week = pd.read_sql("select max(week_start) m from calendar_weeks", eng).m.iloc[0]
    assert set(df.week_start) == {(pd.Timestamp(last_week) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")}
    assert np.isfinite(df.y_pred).all() and (df.y_pred >= 0).all()
    assert (cfg.data_dir / "artifacts" / "next_week_forecast.csv").exists()


def test_publish_uploads_artifacts_to_s3(aws_env, cfg, raw_csv):
    from moto import mock_aws
    from retailsense.pipeline import tasks
    from retailsense.storage.s3 import S3Store

    tasks.ingest(cfg, raw_csv)
    tasks.build_features_task(cfg)
    tasks.train(cfg, fast=True)
    tasks.forecast_next_week(cfg, fast=True)
    with mock_aws():
        store = S3Store(cfg.s3_bucket)
        keys = tasks.publish(cfg, store, run_date="2024-01-01")
        listed = store.list("runs/2024-01-01/")
    assert {"runs/2024-01-01/experiments.md", "runs/2024-01-01/summary.json",
            "runs/2024-01-01/next_week_forecast.csv"} <= set(listed)
    assert set(keys) == set(listed)


def test_run_all_end_to_end(aws_env, cfg, raw_csv):
    from moto import mock_aws
    from retailsense.pipeline import tasks
    from retailsense.storage.s3 import S3Store

    with mock_aws():
        res = tasks.run_all(cfg, raw_csv, store=S3Store(cfg.s3_bucket), fast=True, run_date="2024-02-02")
        assert res["published"] and res["summary"]["best_model"]


# ------------------------------------------------------------------ Airflow DAG (airflow stubbed: not installable on Windows)
def _load_dag_with_stub():
    recorded = {"dag": None, "tasks": {}, "edges": []}

    class Op:
        def __init__(self, task_id, python_callable=None, op_kwargs=None, **kw):
            self.task_id, self.python_callable, self.kw = task_id, python_callable, kw
            recorded["tasks"][task_id] = self
            if recorded["dag"] is not None:
                recorded["dag"].task_ids.append(task_id)

        def __rshift__(self, other):
            for o in (other if isinstance(other, list) else [other]):
                recorded["edges"].append((self.task_id, o.task_id))
            return other

    class DAG:
        def __init__(self, dag_id, **kw):
            self.dag_id, self.kw, self.task_ids = dag_id, kw, []

        def __enter__(self):
            recorded["dag"] = self
            return self

        def __exit__(self, *a):
            return False

    airflow = types.ModuleType("airflow")
    airflow.DAG = DAG
    ops = types.ModuleType("airflow.operators")
    py = types.ModuleType("airflow.operators.python")
    py.PythonOperator = Op
    airflow.operators = ops
    ops.python = py
    saved = {k: sys.modules.get(k) for k in ("airflow", "airflow.operators", "airflow.operators.python")}
    sys.modules.update({"airflow": airflow, "airflow.operators": ops, "airflow.operators.python": py})
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location("retailsense_weekly_dag", ROOT / "dags" / "retailsense_weekly.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return recorded


def test_dag_defines_expected_task_chain():
    rec = _load_dag_with_stub()
    assert rec["dag"].dag_id == "retailsense_weekly"
    assert rec["dag"].kw.get("schedule") in ("@weekly", "0 6 * * 1")
    chain = ["ingest", "validate", "build_features", "train", "forecast", "publish_s3"]
    assert set(rec["tasks"]) == set(chain)
    assert [(a, b) for a, b in zip(chain, chain[1:])] == rec["edges"]
    assert all(callable(t.python_callable) for t in rec["tasks"].values())
    assert rec["dag"].kw.get("catchup") is False
