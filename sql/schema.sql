-- Portable DDL (SQLite + PostgreSQL). Dates are stored as ISO-8601 text.
CREATE TABLE IF NOT EXISTS raw_sales (
    invoice        TEXT NOT NULL,
    stock_code     TEXT NOT NULL,
    description    TEXT,
    quantity       INTEGER NOT NULL,
    invoice_date   TEXT NOT NULL,
    week_start     TEXT NOT NULL,
    price          DOUBLE PRECISION NOT NULL,
    customer_id    TEXT,
    country        TEXT,
    is_return      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS calendar_weeks (
    week_start TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS features (
    sku        TEXT NOT NULL,
    week_start TEXT NOT NULL,
    sales_qty  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id     TEXT NOT NULL,
    model      TEXT,
    sku        TEXT NOT NULL,
    week_start TEXT NOT NULL,
    split      TEXT NOT NULL,
    y_true     DOUBLE PRECISION,
    y_pred     DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id      TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    family      TEXT NOT NULL,
    params_json TEXT,
    split       TEXT NOT NULL,
    level       TEXT NOT NULL,
    mape        DOUBLE PRECISION,
    wape        DOUBLE PRECISION,
    rmse        DOUBLE PRECISION,
    n_rows      INTEGER,
    created_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_raw_sales_sku_week ON raw_sales (stock_code, week_start);
