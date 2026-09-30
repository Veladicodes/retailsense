import pandas as pd
import pytest
from sqlalchemy import inspect, text


def test_clean_transactions_filters_junk_and_flags_returns(raw_tx):
    from retailsense.data.ingest import clean_transactions

    df = clean_transactions(raw_tx)
    assert "POST" not in set(df.stock_code)
    assert (df.price > 0).all()
    assert df.is_return.sum() == 1
    assert (df.loc[df.is_return == 1, "quantity"] < 0).all()
    assert (df.loc[df.is_return == 0, "quantity"] > 0).all()


def test_week_start_is_monday(raw_tx):
    from retailsense.data.ingest import clean_transactions

    df = clean_transactions(raw_tx)
    assert (pd.to_datetime(df.week_start).dt.dayofweek == 0).all()


def test_init_schema_creates_tables(engine):
    from retailsense.data.db import init_schema

    init_schema(engine)
    tables = set(inspect(engine).get_table_names())
    assert {"raw_sales", "calendar_weeks", "features", "forecasts", "model_runs"} <= tables


def test_load_raw_sales_roundtrip(engine, raw_tx):
    from retailsense.data.db import init_schema
    from retailsense.data.ingest import clean_transactions, load_raw_sales

    init_schema(engine)
    df = clean_transactions(raw_tx)
    n = load_raw_sales(engine, df)
    with engine.connect() as c:
        assert c.execute(text("select count(*) from raw_sales")).scalar() == n == len(df)


def test_calendar_excludes_partial_weeks(engine, raw_tx):
    from retailsense.data.db import init_schema
    from retailsense.data.ingest import clean_transactions, load_raw_sales, build_calendar

    init_schema(engine)
    df = clean_transactions(raw_tx)
    load_raw_sales(engine, df)
    cal = build_calendar(engine)
    weeks = pd.to_datetime(pd.Series(cal))
    last_tx = pd.to_datetime(df.invoice_date).max()
    assert (weeks + pd.Timedelta(days=6) <= last_tx.normalize() + pd.Timedelta(days=6)).all()
    assert (weeks.diff().dropna() == pd.Timedelta(days=7)).all()


@pytest.fixture
def loaded(engine, raw_tx):
    from retailsense.data.db import init_schema
    from retailsense.data.ingest import clean_transactions, load_raw_sales, build_calendar

    init_schema(engine)
    load_raw_sales(engine, clean_transactions(raw_tx))
    build_calendar(engine)
    return engine


def test_weekly_panel_is_dense_and_zero_filled(loaded):
    from retailsense.data.queries import weekly_panel

    p = weekly_panel(loaded, top_n=3, min_weeks=10)
    assert set(p.columns) >= {"sku", "week_start", "sales_qty", "returns_qty", "avg_price", "n_invoices", "n_customers"}
    counts = p.groupby("sku").size()
    assert counts.nunique() == 1  # every sku has every calendar week
    assert p.sales_qty.notna().all() and (p.sales_qty >= 0).all()
    assert p.returns_qty.notna().all()


def test_weekly_panel_top_n_limits_skus(loaded):
    from retailsense.data.queries import weekly_panel

    p = weekly_panel(loaded, top_n=2, min_weeks=10)
    assert p.sku.nunique() == 2


def test_weekly_panel_aggregates_match_pandas(loaded, raw_tx):
    from retailsense.data.ingest import clean_transactions
    from retailsense.data.queries import weekly_panel

    p = weekly_panel(loaded, top_n=3, min_weeks=10)
    c = clean_transactions(raw_tx)
    expected = c[c.is_return == 0].groupby(["stock_code", "week_start"]).quantity.sum()
    row = p[(p.sku == "B2")].iloc[10]
    assert row.sales_qty == expected[("B2", str(row.week_start)[:10])]


def test_min_weeks_filters_sparse_skus(loaded):
    from retailsense.data.queries import weekly_panel

    assert weekly_panel(loaded, top_n=10, min_weeks=10_000).empty


def test_split_boundaries_from_sql(loaded):
    from retailsense.data.queries import split_boundaries

    b = split_boundaries(loaded, holdout_weeks=5, validation_weeks=4)
    assert b.val_start < b.test_start
    assert b.train_end < b.val_start
    assert b.n_test == 5 and b.n_val == 4


def test_split_boundaries_require_enough_weeks(loaded):
    from retailsense.data.queries import split_boundaries

    with pytest.raises(ValueError):
        split_boundaries(loaded, holdout_weeks=60, validation_weeks=60)
