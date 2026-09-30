"""Raw UCI Online Retail II transactions -> cleaned lines in SQL."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

# StockCodes that are fees/adjustments rather than products.
_NON_PRODUCT = r"^(?:POST|DOT|M|BANK CHARGES|AMAZONFEE|ADJUST\d*|B|S|CRUK|D|TEST\d*|GIFT.*|C2|PADS)$"

_COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_date",
            "week_start", "price", "customer_id", "country", "is_return"]


def _fmt_customer(v) -> str | None:
    if pd.isna(v):
        return None
    f = float(v)
    return str(int(f)) if f.is_integer() else str(v)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={
        "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
        "Customer ID": "customer_id", "Country": "country",
    }).copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df = df[~df["stock_code"].str.match(_NON_PRODUCT, case=False)]
    df = df[df["price"] > 0]
    df["is_return"] = df["invoice"].str.upper().str.startswith("C").astype(int)
    keep = ((df.is_return == 1) & (df.quantity < 0)) | ((df.is_return == 0) & (df.quantity > 0))
    df = df[keep].copy()
    ts = pd.to_datetime(df["invoice_date"])
    df["week_start"] = (ts.dt.normalize() - pd.to_timedelta(ts.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    df["invoice_date"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["customer_id"] = df["customer_id"].apply(_fmt_customer)
    return df[_COLUMNS].reset_index(drop=True)


def load_raw_sales(engine: Engine, df: pd.DataFrame, chunk: int = 50_000) -> int:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw_sales"))
    df.to_sql("raw_sales", engine, if_exists="append", index=False, chunksize=chunk)
    return len(df)


def build_calendar(engine: Engine) -> list[str]:
    """Populate calendar_weeks with complete Monday-start weeks only."""
    with engine.connect() as conn:
        lo, hi = conn.execute(text("SELECT MIN(invoice_date), MAX(invoice_date) FROM raw_sales")).one()
    first, last = pd.to_datetime(lo), pd.to_datetime(hi)
    start = first.normalize()
    if start.dayofweek != 0:
        start = start + pd.Timedelta(days=7 - start.dayofweek)  # first full week starts next Monday
    weeks = []
    w = start
    while w + pd.Timedelta(days=6) <= last.normalize():
        weeks.append(w.strftime("%Y-%m-%d"))
        w += pd.Timedelta(days=7)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM calendar_weeks"))
        for wk in weeks:
            conn.execute(text("INSERT INTO calendar_weeks (week_start) VALUES (:w)"), {"w": wk})
    return weeks
