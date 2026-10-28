"""Union of all unit panels per (panel file, asset): consistency across units, calendar, coverage."""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
UNITS = ROOT / "repo" / "units"

frames = []
for u in sorted(UNITS.iterdir()):
    for p in sorted(u.glob("*.parquet")):
        df = pd.read_parquet(p)
        col = "asset" if "asset" in df.columns else "asset_id"
        df = df.rename(columns={col: "asset"})
        df["date"] = pd.to_datetime(df["date"])
        df["unit"] = u.name
        df["panel"] = p.stem
        frames.append(df[["unit", "panel", "asset", "date", "value"]])
allp = pd.concat(frames, ignore_index=True)
print("rows", len(allp), "columns of last df", list(df.columns))

for (panel, asset), g in allp.groupby(["panel", "asset"]):
    piv = g.groupby("date")["value"].agg(["min", "max", "count"])
    incons = piv[(piv["max"] - piv["min"]).abs() > 1e-12]
    dates = piv.index
    wk = (dates.dayofweek >= 5).sum()
    nan = g["value"].isna().sum()
    bd = pd.bdate_range(dates.min(), dates.max())
    missing_bd = len(bd.difference(dates))
    print(f"{panel:20s} {asset:8s} {dates.min().date()} {dates.max().date()} n={len(piv)} "
          f"inconsistent_dates={len(incons)} weekend={wk} nan={nan} missing_bdays={missing_bd}")
    if len(incons) and len(incons) < 2000:
        # which units disagree, sample
        ex = incons.head(3)
        for d in ex.index:
            print("    ", d.date(), g[g["date"] == d][["unit", "value"]].drop_duplicates("value").values.tolist()[:4])

# calendar sample: rates around July 4 2019 and Christmas
r = allp[(allp["panel"] == "rates_daily") & (allp["asset"] == "UST_10Y")].drop_duplicates("date").set_index("date")["value"].sort_index()
print(r["2019-06-28":"2019-07-10"])
f = allp[(allp["panel"] == "g10_fx_daily") & (allp["asset"] == "EUR")].drop_duplicates("date").set_index("date")["value"].sort_index()
print(f["2019-06-28":"2019-07-10"])
k = allp[(allp["panel"] == "factors_daily") & (allp["asset"] == "MKT")].drop_duplicates("date").set_index("date")["value"].sort_index()
print(k["2019-06-28":"2019-07-10"])
allp.to_parquet(ROOT / "all_panels.parquet", index=False)
