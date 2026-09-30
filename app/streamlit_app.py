"""RetailSense Enterprise dashboard. Reads everything from SQL (model_runs / forecasts / raw_sales)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retailsense.config import load_config  # noqa: E402
from retailsense.data.db import get_engine  # noqa: E402
from retailsense.features.registry import FEATURE_NAMES  # noqa: E402

st.set_page_config(page_title="RetailSense Enterprise", page_icon="📊", layout="wide")
st.title("RetailSense Enterprise — Retail Demand Forecasting")
st.caption("Python + SQL • 67 leak-free features • XGBoost / LightGBM / ensembles • Airflow + S3 • Docker")


@st.cache_data(ttl=60, show_spinner=False)
def load_runs(db_url: str) -> pd.DataFrame | None:
    try:
        with get_engine(db_url).connect() as c:
            df = pd.read_sql(text("SELECT * FROM model_runs"), c)
    except Exception:
        return None
    return df if not df.empty else None


@st.cache_data(ttl=60, show_spinner=False)
def load_forecasts(db_url: str, run_id: str) -> pd.DataFrame:
    with get_engine(db_url).connect() as c:
        df = pd.read_sql(text("SELECT * FROM forecasts WHERE run_id = :r"), c, params={"r": run_id})
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def select_models(runs: pd.DataFrame) -> tuple[str, str]:
    """Replicates the pipeline's rule: lowest validation SKU-level WAPE, separately for models and baselines."""
    val = runs[(runs.split == "val") & (runs.level == "sku")]
    best = val[val.family != "baseline"].set_index("model_name")["wape"].idxmin()
    base = val[val.family == "baseline"].set_index("model_name")["wape"].idxmin()
    return best, base


cfg = load_config()
runs_all = load_runs(cfg.db_url)
if runs_all is None:
    st.info("No experiment results found yet. Run the pipeline first: `make experiments` "
            "(or trigger the `retailsense_weekly` DAG in Airflow).")
    st.stop()

latest = runs_all.sort_values("created_at").run_id.iloc[-1]
runs = runs_all[runs_all.run_id == latest]
best, base = select_models(runs)


def metric(model: str, level: str, key: str) -> float:
    r = runs[(runs.model_name == model) & (runs.split == "test") & (runs.level == level)]
    return float(r[key].iloc[0])


n_models = runs[runs.family != "baseline"].model_name.nunique()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline MAPE (SKU-week)", f"{metric(base, 'sku', 'mape'):.1f}%", help=f"Best baseline: {base}")
c2.metric("Best model MAPE (SKU-week)", f"{metric(best, 'sku', 'mape'):.1f}%",
          delta=f"{metric(best, 'sku', 'mape') - metric(base, 'sku', 'mape'):+.1f} pts", delta_color="inverse",
          help=f"Selected on validation: {best}")
c3.metric("Features", f"{len(FEATURE_NAMES)}")
c4.metric("Model configs compared", f"{n_models}")

w1, w2, w3 = st.columns(3)
w1.metric("SKU-level WAPE: baseline", f"{metric(base, 'sku', 'wape'):.1f}%")
w2.metric("SKU-level WAPE: best model", f"{metric(best, 'sku', 'wape'):.1f}%")
w3.metric("Weekly total MAPE: baseline vs model", f"{metric(base, 'total', 'mape'):.1f}% / {metric(best, 'total', 'mape'):.1f}%")
st.caption("Holdout = the last weeks of history, never used for training or model selection. "
           "Numbers are measured by the pipeline, not entered by hand.")

tab_models, tab_fc, tab_data = st.tabs(["Model comparison", "Forecast explorer", "Data & pipeline"])

with tab_models:
    test = runs[(runs.split == "test") & (runs.level == "sku")][["model_name", "family", "mape", "wape", "rmse"]]
    val = runs[(runs.split == "val") & (runs.level == "sku")][["model_name", "wape"]].rename(columns={"wape": "val_wape"})
    table = test.merge(val, on="model_name").sort_values("val_wape").rename(columns={
        "model_name": "model", "mape": "holdout_mape", "wape": "holdout_wape", "rmse": "holdout_rmse"})
    st.dataframe(table.round(1), use_container_width=True, hide_index=True)
    st.bar_chart(table.set_index("model")["holdout_wape"])

with tab_fc:
    fc = load_forecasts(cfg.db_url, latest)
    hist = fc[fc.split == "test"]
    if hist.empty:
        st.info("No holdout forecasts stored for this run.")
    else:
        sku = st.selectbox("SKU", sorted(hist.sku.unique()))
        s = hist[hist.sku == sku]
        chart = s.pivot(index="week_start", columns="model", values="y_pred")
        chart["actual"] = s.drop_duplicates("week_start").set_index("week_start")["y_true"]
        st.line_chart(chart)
        future = fc[fc.split == "future"]
        if not future.empty:
            row = future[future.sku == sku]
            if not row.empty:
                st.metric(f"Next-week forecast ({row.week_start.iloc[0]:%Y-%m-%d})", f"{row.y_pred.iloc[0]:,.0f} units")

with tab_data:
    with get_engine(cfg.db_url).connect() as c:
        stats = {t: c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() for t in
                 ("raw_sales", "calendar_weeks", "model_runs", "forecasts")}
    st.write("Rows per SQL table")
    st.dataframe(pd.DataFrame({"table": list(stats), "rows": list(stats.values())}), hide_index=True)
    st.write(f"Latest run: `{latest}` — feature registry: {len(FEATURE_NAMES)} features")
    st.code(", ".join(FEATURE_NAMES), language="text")
