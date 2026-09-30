import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"


def _source_files():
    files = [p for p in LEGACY.rglob("*.py")]
    files += [p for p in LEGACY.rglob("*.ipynb")]
    files += [p for p in (LEGACY / "README").rglob("*.md")]
    return files


def test_config_defaults_are_relative_and_overridable(monkeypatch):
    from retailsense import config

    monkeypatch.setenv("RETAILSENSE_DATA_DIR", "somewhere")
    monkeypatch.setenv("RETAILSENSE_DB_URL", "sqlite:///x.db")
    cfg = config.load_config()
    assert cfg.data_dir.name == "somewhere"
    assert cfg.db_url == "sqlite:///x.db"
    assert not str(cfg.data_dir).startswith("F:")


def test_config_has_s3_settings(monkeypatch):
    from retailsense import config

    monkeypatch.setenv("RETAILSENSE_S3_BUCKET", "b")
    monkeypatch.setenv("RETAILSENSE_S3_ENDPOINT", "http://minio:9000")
    cfg = config.load_config()
    assert cfg.s3_bucket == "b"
    assert cfg.s3_endpoint == "http://minio:9000"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_hardcoded_drive_paths(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    assert not re.search(r"F:\\?RetailSense_Lite", text, flags=re.I), path


def test_legacy_app_defines_inventory_variables_it_uses():
    tree = ast.parse((LEGACY / "app.py").read_text(encoding="utf-8"))
    assigned = {
        t.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    for name in ("safety_stock_factor", "lead_time_demand"):
        assert name in assigned, f"{name} used in app.py but never assigned"


def test_advanced_forecasting_has_single_run_hybrid_forecast():
    tree = ast.parse((LEGACY / "utils" / "advanced_forecasting.py").read_text(encoding="utf-8"))
    defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert defs.count("run_hybrid_forecast") == 1


def test_scratch_script_uses_valid_keyword():
    text = (LEGACY / "hai.py").read_text(encoding="utf-8")
    assert "horizon=90" not in text


def test_no_deprecated_fillna_method_in_advanced_forecasting():
    text = (LEGACY / "utils" / "advanced_forecasting.py").read_text(encoding="utf-8")
    assert "fillna(method=" not in text
