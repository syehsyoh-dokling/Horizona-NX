"""Inventory of the 22 short-horizon practice cards (ids ending -ps1).

Analysis-only script (never shipped in the agent image). Prints, per card: family, panel,
assets, horizons, as-of, last observed value per asset, realised values and target dates,
plus the trailing-300 daily step sd (the M0 width input) so widths can be judged later.
"""
import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd

PUB = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    "C:/Users/Saifuddin/Documents/Agenthon-2026/t2work/public")
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(
    "C:/Users/Saifuddin/Documents/Agenthon-2026/t2work/sim/inventory_ps1.json")

rows = []
for unit in sorted(PUB.glob("units/*-ps1")):
    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    uid = card["task"]["id"]
    fam = card["metadata"]["category"]
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    tg = card["targets"]
    assets = list(tg["asset_ids"])
    horizons = [int(h) for h in tg["horizons"]]
    panel_id = card["panels"]["panel_ids"][0]
    panel = pd.read_parquet(unit / f"{panel_id}.parquet")
    ans = pd.read_parquet(PUB / "practice-answers" / f"{uid}.parquet")
    per_asset = {}
    for a in assets:
        s = panel[panel["asset"].astype(str) == a]
        s = pd.Series(s["value"].astype(float).to_numpy(), index=pd.to_datetime(s["date"]))
        s = s.sort_index()
        s = s[s.index <= asof]
        last_date = s.index[-1]
        hist = s.iloc[-300:]
        sd300 = float(np.std(np.diff(hist.to_numpy()), ddof=1))
        sd60 = float(np.std(np.diff(s.iloc[-61:].to_numpy()), ddof=1))
        sd20 = float(np.std(np.diff(s.iloc[-21:].to_numpy()), ddof=1))
        real = {int(r.horizon): (float(r.value), str(r.target_date))
                for r in ans[ans["asset"] == a].itertuples()}
        per_asset[a] = {
            "last_date": str(last_date.date()), "last": float(s.iloc[-1]),
            "sd300": sd300, "sd60": sd60, "sd20": sd20,
            "realised": {str(h): {"value": v, "date": d, "z300": (v - float(s.iloc[-1])) / (sd300 * np.sqrt(h))}
                         for h, (v, d) in real.items()},
        }
    rows.append({"unit": uid, "family": fam, "panel": panel_id, "asof": str(asof.date()),
                 "weekday": asof.day_name(), "assets": assets, "horizons": horizons,
                 "n_cells": len(assets) * len(horizons), "per_asset": per_asset,
                 "panel_cols": list(panel.columns), "panel_end": str(pd.to_datetime(panel["date"]).max().date())})

OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
print(f"{'unit':46s} {'fam':5s} {'panel':16s} {'asof':10s} {'wd':3s} hor        assets")
for r in rows:
    print(f"{r['unit']:46s} {r['family']:5s} {r['panel']:16s} {r['asof']:10s} {r['weekday'][:3]:3s} "
          f"{str(r['horizons']):10s} {','.join(r['assets'])}")
print()
print("per asset: last | realised(h)=value [z vs sd300*sqrt(h)] | sd300 sd60 sd20")
for r in rows:
    for a, d in r["per_asset"].items():
        rs = " ".join(f"h{h}={x['value']:.6g}[{x['z300']:+.2f}]" for h, x in d["realised"].items())
        print(f"  {r['unit']:44s} {a:7s} last={d['last']:.6g} {rs}  sd300={d['sd300']:.4g} sd60={d['sd60']:.4g} sd20={d['sd20']:.4g}")
print("panel cols:", rows[0]["panel_cols"])
