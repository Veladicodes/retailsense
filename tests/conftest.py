import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine


def make_transactions(n_weeks=80, skus=("A1", "B2", "C3"), seed=0, start="2010-01-04"):
    """Synthetic raw transaction lines shaped like UCI Online Retail II."""
    rng = np.random.default_rng(seed)
    rows = []
    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    inv = 1000
    for si, sku in enumerate(skus):
        for wi, w in enumerate(weeks):
            base = 50 + 10 * si + 15 * np.sin(2 * np.pi * wi / 52)
            for d in range(3):
                inv += 1
                qty = max(1, int(rng.poisson(base / 3)))
                rows.append((str(inv), sku, f"item {sku}", qty, w + pd.Timedelta(days=d, hours=10),
                             round(2.0 + si + 0.05 * rng.standard_normal(), 2), 10000 + (inv % 7), "United Kingdom"))
    # one return line and one junk line
    rows.append(("C9001", "A1", "item A1", -3, weeks[5] + pd.Timedelta(days=1), 2.0, 10001, "United Kingdom"))
    rows.append(("9002", "POST", "postage", 1, weeks[5], 18.0, 10002, "United Kingdom"))
    rows.append(("9003", "A1", "bad price", 5, weeks[6], 0.0, 10003, "United Kingdom"))
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])


@pytest.fixture
def raw_tx():
    return make_transactions()


@pytest.fixture
def engine():
    return create_engine("sqlite://")
