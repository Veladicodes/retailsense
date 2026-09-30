"""SQL-backed reads: demand panel and train/validation/test boundaries."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from retailsense.data.db import read_sql_file


def weekly_panel(engine: Engine, top_n: int = 100, min_weeks: int = 90) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(read_sql_file("weekly_panel.sql")), conn,
                         params={"top_n": top_n, "min_weeks": min_weeks})
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: pd.Timestamp  # last training week
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    n_val: int
    n_test: int


def split_boundaries(engine: Engine, holdout_weeks: int, validation_weeks: int) -> SplitBoundaries:
    with engine.connect() as conn:
        weeks = [pd.Timestamp(r[0]) for r in conn.execute(text(read_sql_file("split_weeks.sql")))]
    need = holdout_weeks + validation_weeks + 26  # keep at least half a year of training history
    if len(weeks) < need:
        raise ValueError(f"Need at least {need} weeks of history, found {len(weeks)}")
    test_start = weeks[holdout_weeks - 1]
    val_start = weeks[holdout_weeks + validation_weeks - 1]
    train_end = weeks[holdout_weeks + validation_weeks]
    return SplitBoundaries(train_end, val_start, test_start, validation_weeks, holdout_weeks)
