import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_transactions


# ---------------------------------------------------------------- metrics
def test_mape_ignores_zero_actuals():
    from retailsense.models.metrics import mape

    y = np.array([100.0, 0.0, 50.0])
    p = np.array([110.0, 5.0, 40.0])
    assert mape(y, p) == pytest.approx((10 / 100 + 10 / 50) / 2 * 100)


def test_wape_and_rmse():
    from retailsense.models.metrics import rmse, wape

    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    assert wape(y, p) == pytest.approx(7 / 60 * 100)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))


def test_total_level_mape_aggregates_across_skus_per_week():
    from retailsense.models.metrics import evaluate

    df = pd.DataFrame({
        "sku": ["a", "b", "a", "b"],
        "week_start": pd.to_datetime(["2020-01-06", "2020-01-06", "2020-01-13", "2020-01-13"]),
        "y_true": [100.0, 100.0, 100.0, 100.0],
        "y_pred": [150.0, 50.0, 110.0, 90.0],  # errors cancel in the total
    })
    sku = evaluate(df, level="sku")
    tot = evaluate(df, level="total")
    assert sku["mape"] == pytest.approx((50 + 50 + 10 + 10) / 4)
    assert tot["mape"] == pytest.approx(0.0)


# ---------------------------------------------------------------- baselines
def _features(n_weeks=110, n_skus=6):
    from retailsense.data.db import init_schema
    from retailsense.data.ingest import build_calendar, clean_transactions, load_raw_sales
    from retailsense.data.queries import split_boundaries, weekly_panel
    from retailsense.features.registry import assign_split, build_features
    from sqlalchemy import create_engine

    eng = create_engine("sqlite://")
    init_schema(eng)
    skus = tuple(f"S{i}" for i in range(n_skus))
    load_raw_sales(eng, clean_transactions(make_transactions(n_weeks=n_weeks, skus=skus)))
    build_calendar(eng)
    panel = weekly_panel(eng, top_n=n_skus, min_weeks=20)
    b = split_boundaries(eng, holdout_weeks=8, validation_weeks=8)
    return assign_split(build_features(panel), b)


@pytest.fixture(scope="module")
def feats():
    return _features()


def test_baselines_are_defined_from_past_features(feats):
    from retailsense.models.baselines import BASELINES, predict_baseline

    assert {"naive_last_week", "seasonal_naive_52", "moving_avg_4", "ewm_03"} <= set(BASELINES)
    row = feats[feats.split == "test"]
    assert predict_baseline("naive_last_week", row).equals(row["lag_1"])
    p = predict_baseline("seasonal_naive_52", row)
    assert p.notna().all()  # falls back to last week when lag_52 missing


# ---------------------------------------------------------------- grid
def test_at_least_15_model_configurations():
    from retailsense.models.grid import build_configs

    cfgs = build_configs()
    names = [c["name"] for c in cfgs]
    assert len(cfgs) >= 15
    assert len(set(names)) == len(names)
    assert {"xgboost", "lightgbm", "random_forest", "ridge"} <= {c["family"] for c in cfgs}


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "random_forest", "ridge"])
def test_every_family_fits_and_predicts_finite(feats, family):
    from retailsense.features.registry import FEATURE_NAMES
    from retailsense.models.grid import build_configs, fit_predict

    cfg = next(c for c in build_configs(fast=True) if c["family"] == family)
    tr = feats[feats.split == "train"].dropna(subset=["lag_1"])
    te = feats[feats.split == "val"]
    pred = fit_predict(cfg, tr, te, FEATURE_NAMES)
    assert len(pred) == len(te)
    assert np.isfinite(pred).all() and (pred >= 0).all()


def test_configs_include_median_objective_variants():
    from retailsense.models.grid import build_configs

    l1 = [c for c in build_configs() if c["params"].get("objective") in ("l1", "reg:absoluteerror")]
    assert {c["family"] for c in l1} == {"xgboost", "lightgbm"}
    assert len(build_configs()) >= 20


def test_configs_include_raw_log_and_ratio_targets():
    from retailsense.models.grid import build_configs

    assert {"raw", "log", "ratio"} <= {c["target"] for c in build_configs()}


def test_ratio_target_is_scale_invariant(feats):
    """Scaling one SKU's demand by 10x must scale its ratio-model predictions ~10x, not leave them unchanged."""
    from retailsense.features.registry import FEATURE_NAMES
    from retailsense.models.grid import build_configs, fit_predict

    cfg = next(c for c in build_configs(fast=True) if c["target"] == "ratio" and c["family"] == "ridge") \
        if any(c["target"] == "ratio" and c["family"] == "ridge" for c in build_configs(fast=True)) \
        else next(c for c in build_configs(fast=True) if c["target"] == "ratio")
    tr = feats[feats.split == "train"].dropna(subset=["lag_1"]).copy()
    te = feats[feats.split == "val"].copy()
    base = fit_predict(cfg, tr, te, FEATURE_NAMES)
    # 10x every level-type quantity of one sku (train + predict rows)
    sku = tr.sku.iloc[0]
    level_cols = [c for c in FEATURE_NAMES if c.startswith(("lag_", "roll_mean", "roll_min", "roll_max", "roll_std", "ewm", "exp_mean", "exp_max", "exp_std", "seas52"))]
    for d in (tr, te):
        m = d.sku == sku
        d.loc[m, level_cols + ["sales_qty"]] = d.loc[m, level_cols + ["sales_qty"]] * 10
    scaled = fit_predict(cfg, tr, te, FEATURE_NAMES)
    m = (te.sku == sku).to_numpy()
    assert scaled[m].mean() > 5 * base[m].mean()


# ---------------------------------------------------------------- experiment protocol
@pytest.fixture(scope="module")
def result(feats):
    from retailsense.models.experiment import run_experiments

    return run_experiments(feats, fast=True)


def test_experiment_reports_every_config_and_baseline(result):
    from retailsense.models.grid import build_configs

    names = set(result.runs.model)
    assert {c["name"] for c in build_configs()} <= names
    assert {"naive_last_week", "seasonal_naive_52", "moving_avg_4", "ewm_03"} <= names
    assert {"val", "test"} == set(result.runs.split)
    assert {"sku", "total"} == set(result.runs.level)


def test_model_selection_uses_validation_only(feats, result):
    from retailsense.models.experiment import run_experiments

    val = result.runs[(result.runs.split == "val") & (result.runs.level == "sku") & (result.runs.family != "baseline")]
    assert result.summary["best_model"] == val.loc[val.wape.idxmin(), "model"]
    base = result.runs[(result.runs.split == "val") & (result.runs.level == "sku") & (result.runs.family == "baseline")]
    assert result.summary["baseline_model"] == base.loc[base.wape.idxmin(), "model"]
    # corrupt the test actuals: the chosen model must not change
    bad = feats.copy()
    bad.loc[bad.split == "test", "sales_qty"] = 123456.0
    r2 = run_experiments(bad, fast=True)
    assert r2.summary["best_model"] == result.summary["best_model"]


def test_top3_ensemble_is_evaluated_and_uses_validation_ranking(result):
    runs = result.runs
    assert "ensemble_top3" in set(runs.model)
    params = runs[(runs.model == "ensemble_top3") & (runs.split == "val") & (runs.level == "sku")].params_json.iloc[0]
    members = json.loads(params)["members"]
    val = runs[(runs.split == "val") & (runs.level == "sku") & (~runs.family.isin(["baseline", "ensemble"]))]
    assert members == list(val.sort_values("wape").model.head(3))


def test_summary_contains_baseline_and_improvement(result):
    s = result.summary
    for k in ("baseline_model", "baseline_mape_sku", "best_model", "best_mape_sku",
              "baseline_mape_total", "best_mape_total", "n_configs", "n_skus", "n_test_weeks"):
        assert k in s
    assert s["n_configs"] >= 15
    best_test = result.runs[(result.runs.model == s["best_model"]) & (result.runs.split == "test") & (result.runs.level == "sku")]
    assert s["best_mape_sku"] == pytest.approx(best_test.mape.iloc[0])


def test_predictions_cover_test_rows(feats, result):
    p = result.predictions
    assert set(p.columns) >= {"sku", "week_start", "split", "y_true", "y_pred", "model"}
    n_test = (feats.split == "test").sum()
    assert len(p[(p.split == "test") & (p.model == result.summary["best_model"])]) == n_test


# ---------------------------------------------------------------- logging + report
def test_runs_are_logged_to_sql(engine, result):
    from retailsense.data.db import init_schema
    from retailsense.models.experiment import log_runs
    from sqlalchemy import text

    init_schema(engine)
    run_id = log_runs(engine, result)
    with engine.connect() as c:
        n = c.execute(text("select count(*) from model_runs where run_id=:r"), {"r": run_id}).scalar()
        nf = c.execute(text("select count(*) from forecasts where run_id=:r"), {"r": run_id}).scalar()
    assert n == len(result.runs) and nf > 0


def test_logged_forecasts_identify_their_model(engine, result):
    import pandas as pd
    from retailsense.data.db import init_schema
    from retailsense.models.experiment import log_runs

    init_schema(engine)
    run_id = log_runs(engine, result)
    fc = pd.read_sql(f"select * from forecasts where run_id='{run_id}'", engine)
    assert "model" in fc.columns
    assert set(fc.model) == {result.summary["best_model"], result.summary["baseline_model"]}


def test_report_is_written_with_real_numbers(tmp_path, result):
    from retailsense.reporting import write_report

    md = tmp_path / "experiments.md"
    js = tmp_path / "summary.json"
    write_report(result, md, js)
    text = md.read_text(encoding="utf-8")
    s = json.loads(js.read_text(encoding="utf-8"))
    assert f"{s['best_mape_sku']:.1f}%" in text
    assert f"{s['baseline_mape_sku']:.1f}%" in text
    assert "validation" in text.lower() and "holdout" in text.lower()
