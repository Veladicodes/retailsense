import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from tests.conftest import make_transactions

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ Docker
def test_dockerfile_installs_requirements_and_runs_streamlit():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11" in text
    assert "requirements.txt" in text
    assert "streamlit" in text and "app/streamlit_app.py" in text


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_has_all_services(compose):
    assert {"postgres", "minio", "airflow-init", "airflow-webserver", "airflow-scheduler", "streamlit"} <= set(compose["services"])


def test_compose_wires_sql_and_s3(compose):
    for name in ("airflow-scheduler", "streamlit"):
        env = compose["services"][name]["environment"]
        env = env if isinstance(env, dict) else dict(e.split("=", 1) for e in env)
        assert env["RETAILSENSE_DB_URL"].startswith("postgresql")
        assert env["RETAILSENSE_S3_ENDPOINT"] == "http://minio:9000"
    assert "minio/minio" in compose["services"]["minio"]["image"]
    assert "postgres" in compose["services"]["postgres"]["image"]


def test_compose_mounts_dags_and_exposes_ports(compose):
    sched_vols = " ".join(compose["services"]["airflow-scheduler"]["volumes"])
    assert "./dags" in sched_vols and "./retailsense" in sched_vols
    assert any("8501" in str(p) for p in compose["services"]["streamlit"]["ports"])
    assert any("8080" in str(p) for p in compose["services"]["airflow-webserver"]["ports"])


def test_requirements_pin_core_packages():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("pandas", "xgboost", "lightgbm", "sqlalchemy", "boto3", "streamlit", "scikit-learn"):
        assert pkg in req


def test_makefile_targets():
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("up:", "down:", "test:", "experiments:", "data:"):
        assert target in mk


def test_ci_workflow_runs_pytest():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    steps = json.dumps(wf)
    assert "pytest" in steps and "docker" in steps.lower()


# ------------------------------------------------------------------ Streamlit
@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory):
    from retailsense.config import load_config
    from retailsense.pipeline import tasks

    tmp = tmp_path_factory.mktemp("app")
    raw = tmp / "raw.csv"
    make_transactions(n_weeks=110, skus=tuple(f"S{i}" for i in range(6))).to_csv(raw, index=False)
    cfg = dataclasses.replace(load_config(), data_dir=tmp, db_url=f"sqlite:///{(tmp / 'rs.db').as_posix()}",
                              top_n_skus=6, holdout_weeks=8, validation_weeks=8)
    tasks.run_all(cfg, raw, store=None, fast=True)
    return cfg


def test_streamlit_app_renders_measured_metrics(seeded_db, monkeypatch_module):
    from streamlit.testing.v1 import AppTest

    monkeypatch_module.setenv("RETAILSENSE_DB_URL", seeded_db.db_url)
    monkeypatch_module.setenv("RETAILSENSE_DATA_DIR", str(seeded_db.data_dir))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "RetailSense Enterprise" in at.title[0].value
    labels = {m.label for m in at.metric}
    assert {"Baseline MAPE (SKU-week)", "Best model MAPE (SKU-week)", "Features", "Model configs compared"} <= labels
    feats = next(m for m in at.metric if m.label == "Features")
    assert feats.value == "67"


def test_streamlit_app_handles_empty_database(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RETAILSENSE_DB_URL", f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    monkeypatch.setenv("RETAILSENSE_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("pipeline" in w.value.lower() for w in list(at.info) + list(at.warning))


# ------------------------------------------------------------------ honesty: docs must match measured results
def test_readme_quotes_only_measured_numbers():
    summary = json.loads((ROOT / "reports" / "summary.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "RetailSense Enterprise" in readme
    assert f"{summary['best_mape_sku']:.1f}%" in readme
    assert f"{summary['baseline_mape_sku']:.1f}%" in readme
    assert f"{summary['best_mape_total']:.1f}%" in readme
    assert summary["n_features"] == 67 and summary["n_configs"] >= 15
