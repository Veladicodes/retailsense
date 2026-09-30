import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel(engine, raw_tx):
    from retailsense.data.db import init_schema
    from retailsense.data.ingest import build_calendar, clean_transactions, load_raw_sales
    from retailsense.data.queries import weekly_panel

    init_schema(engine)
    load_raw_sales(engine, clean_transactions(raw_tx))
    build_calendar(engine)
    return weekly_panel(engine, top_n=3, min_weeks=10)


def test_exactly_67_unique_feature_names():
    from retailsense.features.registry import FEATURE_NAMES

    assert len(FEATURE_NAMES) == 67
    assert len(set(FEATURE_NAMES)) == 67


def test_no_target_derived_feature_names():
    from retailsense.features.registry import FEATURE_NAMES

    assert "sales_qty" not in FEATURE_NAMES
    assert not any("sales_stock" in n or "stock_sales" in n for n in FEATURE_NAMES)


def test_build_features_returns_all_columns(panel):
    from retailsense.features.registry import FEATURE_NAMES, build_features

    f = build_features(panel)
    assert {"sku", "week_start", "sales_qty"} <= set(f.columns)
    assert set(FEATURE_NAMES) <= set(f.columns)
    assert len(f) == len(panel)


def test_features_have_no_infinities(panel):
    from retailsense.features.registry import FEATURE_NAMES, build_features

    f = build_features(panel)
    assert not np.isinf(f[FEATURE_NAMES].to_numpy(dtype=float)).any()


def test_lag_and_rolling_values_are_correct(panel):
    from retailsense.features.registry import build_features

    f = build_features(panel)
    s = f[f.sku == "A1"].reset_index(drop=True)
    t = 30
    assert s.loc[t, "lag_1"] == s.loc[t - 1, "sales_qty"]
    assert s.loc[t, "lag_4"] == s.loc[t - 4, "sales_qty"]
    assert s.loc[t, "roll_mean_4"] == pytest.approx(s.loc[t - 4:t - 1, "sales_qty"].mean())
    assert s.loc[t, "roll_max_8"] == s.loc[t - 8:t - 1, "sales_qty"].max()


def test_calendar_features(panel):
    from retailsense.features.registry import build_features

    f = build_features(panel)
    row = f.iloc[10]
    wk = pd.Timestamp(row.week_start)
    assert row.week_of_year == wk.isocalendar().week
    assert row.month == wk.month
    assert -1 <= row.week_sin <= 1
    assert 0 <= row.weeks_to_christmas <= 53


def test_weeks_to_christmas_wraps_to_next_year():
    from retailsense.features.registry import weeks_to_christmas

    s = pd.Series(pd.to_datetime(["2010-12-20", "2010-12-27", "2011-01-03"]))
    w = weeks_to_christmas(s)
    assert w.iloc[0] == pytest.approx(5 / 7, abs=0.01)
    assert w.iloc[1] > 50  # Christmas already passed -> next year's


def test_no_future_leakage_perturbing_target_does_not_change_its_own_row(panel):
    """Changing sales at week t must not change ANY feature at week t (for any sku)."""
    from retailsense.features.registry import FEATURE_NAMES, build_features

    base = build_features(panel)
    t = pd.Timestamp(sorted(panel.week_start.unique())[40])
    pert = panel.copy()
    mask = (pert.sku == "A1") & (pert.week_start == t)
    pert.loc[mask, "sales_qty"] = 1_000_000
    pert.loc[mask, "n_invoices"] = 999
    after = build_features(pert)
    b = base[base.week_start == t].set_index("sku")[FEATURE_NAMES]
    a = after[after.week_start == t].set_index("sku")[FEATURE_NAMES]
    pd.testing.assert_frame_equal(a, b)


def test_perturbation_does_propagate_to_the_next_week(panel):
    from retailsense.features.registry import build_features

    base = build_features(panel)
    weeks = sorted(panel.week_start.unique())
    t, t1 = pd.Timestamp(weeks[40]), pd.Timestamp(weeks[41])
    pert = panel.copy()
    pert.loc[(pert.sku == "A1") & (pert.week_start == t), "sales_qty"] = 1_000_000
    after = build_features(pert)
    get = lambda d: d[(d.sku == "A1") & (d.week_start == t1)].iloc[0].lag_1
    assert get(after) == 1_000_000 and get(base) != 1_000_000


def test_price_features_use_only_past_prices(panel):
    from retailsense.features.registry import build_features

    base = build_features(panel)
    t = pd.Timestamp(sorted(panel.week_start.unique())[40])
    pert = panel.copy()
    pert.loc[(pert.sku == "B2") & (pert.week_start == t), "avg_price"] = 9999.0
    after = build_features(pert)
    cols = ["price_lag_1", "price_lag_4", "price_pct_change_1", "price_rel_mean13", "price_std_8", "price_max_13"]
    sel = lambda d: d[(d.sku == "B2") & (d.week_start == t)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(sel(base), sel(after))


def test_split_assigns_train_val_test(panel):
    from retailsense.data.queries import SplitBoundaries
    from retailsense.features.registry import assign_split, build_features

    f = build_features(panel)
    weeks = sorted(f.week_start.unique())
    b = SplitBoundaries(train_end=pd.Timestamp(weeks[-11]), val_start=pd.Timestamp(weeks[-10]),
                        test_start=pd.Timestamp(weeks[-5]), n_val=5, n_test=5)
    s = assign_split(f, b)
    assert set(s.split.unique()) == {"train", "val", "test"}
    assert s[s.split == "train"].week_start.max() < s[s.split == "val"].week_start.min()
    assert s[s.split == "val"].week_start.max() < s[s.split == "test"].week_start.min()
    assert (s.split == "test").sum() == 5 * f.sku.nunique()
