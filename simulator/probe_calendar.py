"""Dates on which the series inside one panel do not all have a row (per panel, longest copy)."""
import pandas as pd

import t2_core as C

allp = pd.read_parquet(C.HERE / "all_panels.parquet")
allp = allp[allp["unit"] != "t2-EXAMPLE-ust-curve-1m"]
for panel in ("rates_daily", "g10_fx_daily", "factors_daily"):
    g = allp[allp["panel"] == panel].drop_duplicates(["asset", "date"])
    piv = g.pivot(index="date", columns="asset", values="value").sort_index()
    miss = piv[piv.isna().any(axis=1)]
    print(panel, "dates", len(piv), "dates with a missing series", len(miss))
    if panel != "factors_daily":
        print(miss.isna().astype(int).to_string())
    else:
        print(miss.isna().sum())
        cal = piv.dropna().index
        print("factor intersection rows", len(cal))
r = allp[(allp["panel"] == "rates_daily")].drop_duplicates(["asset", "date"]).pivot(index="date", columns="asset", values="value")
f = allp[(allp["panel"] == "g10_fx_daily")].drop_duplicates(["asset", "date"]).pivot(index="date", columns="asset", values="value")
only_r = r.index.difference(f.index)
only_f = f.index.difference(r.index)
print("in rates not fx:", [str(d.date()) for d in only_r])
print("in fx not rates:", [str(d.date()) for d in only_f])
