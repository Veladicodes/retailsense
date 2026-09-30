import pandas as pd
from sqlalchemy import create_engine
e = create_engine("sqlite:///data/retailsense.db")
r = pd.read_sql("select model_name, family, split, level, mape, wape from model_runs", e)
v = r[r.split=="val"].pivot(index="model_name", columns="level", values=["mape","wape"])
v.columns = [f"{a}_{b}" for a,b in v.columns]
print(v.sort_values("wape_sku").round(1).to_string())
