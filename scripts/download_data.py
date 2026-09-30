"""Download UCI Online Retail II (CC BY 4.0) and convert it to data/raw/online_retail_ii.csv."""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OUT = Path("data/raw/online_retail_ii.csv")


def main() -> int:
    if OUT.exists():
        print(f"{OUT} already exists - nothing to do")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} ...")
    payload = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df):,} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
