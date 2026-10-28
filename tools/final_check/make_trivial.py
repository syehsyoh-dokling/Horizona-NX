"""Trivial forecast for one unit: normal draws around the last observation with an arbitrary sd.

Used only to verify rawscore.py against the official scoring/scoring.py CLI.
usage: make_trivial.py <unit_dir(with panels/)> <out_dir> <sd>
"""
import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd

unit = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
sd = float(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
tg = card["targets"]
assets = sorted(str(a) for a in tg["asset_ids"])
horizons = sorted(int(h) for h in tg["horizons"])
asof = pd.Timestamp(card["provenance"]["data_cutoff"])
pq = next((unit / "panels").glob("*.parquet"))
panel = pd.read_parquet(pq)
rng = np.random.default_rng(7)
rows = []
for a in assets:
    s = panel[panel["asset"].astype(str) == a]
    s = pd.Series(s["value"].astype(float).to_numpy(), index=pd.to_datetime(s["date"])).sort_index()
    last = float(s[s.index <= asof].iloc[-1])
    for h in horizons:
        vals = last + sd * np.sqrt(h) * rng.standard_normal(500)
        for d, v in enumerate(vals):
            rows.append((d, a, h, float(v)))
df = pd.DataFrame(rows, columns=["draw", "asset", "horizon", "value"])
df["draw"] = df["draw"].astype("int32")
df["horizon"] = df["horizon"].astype("int32")
df.to_parquet(out / "forecast.parquet", index=False)
(out / "forecast_meta.json").write_text(json.dumps({
    "unit_id": card["task"]["id"], "asof": str(asof.date()), "representation": "samples",
    "asset_ids": list(tg["asset_ids"]), "horizons": [int(h) for h in tg["horizons"]],
    "n_draws": 500, "target": tg["target_type"]}, indent=2))
(out / "forecast_rationale.md").write_text("# trivial\n\nNormal draws about the last value; scorer check only.\n")
print("wrote", out)
