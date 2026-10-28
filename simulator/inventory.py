"""Inventory of the T2 units: split, family, grid, as-of, panels, last obs per target asset."""
import json
import pathlib
import tomllib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
UNITS = ROOT / "repo" / "units"

rows = []
for u in sorted(UNITS.iterdir()):
    card = tomllib.loads((u / "card.toml").read_text(encoding="utf-8"))
    tgt = card["targets"]
    asof = card["provenance"]["data_cutoff"]
    spec = {}
    if (u / "forecast_spec.json").is_file():
        spec = json.loads((u / "forecast_spec.json").read_text(encoding="utf-8"))
    panels = sorted(p.name for p in u.glob("*.parquet"))
    info = {}
    for p in panels:
        df = pd.read_parquet(u / p)
        col = "asset" if "asset" in df.columns else "asset_id"
        df["date"] = pd.to_datetime(df["date"])
        for a in tgt["asset_ids"]:
            s = df[df[col].astype(str) == a]
            if len(s) and a not in info:
                info[a] = (p, str(s["date"].max().date()), int(len(s)))
    rows.append({
        "unit": u.name, "id": card["task"]["id"], "split": card["task"].get("split"),
        "asof": asof, "ttype": tgt.get("target_type"), "freq": tgt.get("target_frequency"),
        "assets": tgt["asset_ids"], "horizons": tgt["horizons"],
        "target_dates": tgt.get("target_dates"),
        "obs_periods": (spec.get("targets") or {}).get("observation_periods"),
        "spec_target_dates": (spec.get("targets") or {}).get("target_dates"),
        "panels": panels, "info": info,
    })

json.dump(rows, open(ROOT / "inventory.json", "w"), indent=1)
df = pd.DataFrame(rows)
print(df["split"].value_counts())
print(df.groupby(["split", df["unit"].str.extract(r"t2-(F\d|EXAMPLE)")[0]]).size())
print(pd.Series([p for r in rows for p in r["panels"]]).value_counts())
for r in rows:
    print(r["unit"], r["split"], r["asof"], r["ttype"], r["freq"], r["assets"], r["horizons"],
          r["obs_periods"], r["target_dates"], r["spec_target_dates"],
          {a: v[0][:6] + ":" + v[1] for a, v in r["info"].items()})
