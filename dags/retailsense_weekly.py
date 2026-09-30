"""Weekly RetailSense pipeline: SQL ingest -> validation -> features -> model grid -> forecast -> S3."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from retailsense.config import load_config
from retailsense.pipeline import tasks
from retailsense.storage.s3 import S3Store

RAW_PATH = os.getenv("RETAILSENSE_RAW_PATH", "data/raw/online_retail_ii.csv")


def _ingest(**_):
    return tasks.ingest(load_config(), RAW_PATH)


def _validate(**_):
    return tasks.validate(load_config())


def _build_features(**_):
    return tasks.build_features_task(load_config())


def _train(**_):
    return tasks.train(load_config())["run_id"]


def _forecast(**_):
    return tasks.forecast_next_week(load_config())


def _publish_s3(run_date=None, **_):
    cfg = load_config()
    return tasks.publish(cfg, S3Store(cfg.s3_bucket, cfg.s3_endpoint), run_date=run_date)


default_args = {"owner": "retailsense", "retries": 2, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="retailsense_weekly",
    description="Weekly demand-forecasting pipeline (SQL + XGBoost/LightGBM + S3)",
    schedule="0 6 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["retail", "forecasting"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    validate = PythonOperator(task_id="validate", python_callable=_validate)
    build_features = PythonOperator(task_id="build_features", python_callable=_build_features)
    train = PythonOperator(task_id="train", python_callable=_train)
    forecast = PythonOperator(task_id="forecast", python_callable=_forecast)
    publish_s3 = PythonOperator(task_id="publish_s3", python_callable=_publish_s3, op_kwargs={"run_date": "{{ ds }}"})

    ingest >> validate >> build_features >> train >> forecast >> publish_s3
