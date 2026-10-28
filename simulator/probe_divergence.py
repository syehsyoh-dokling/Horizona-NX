"""Why does t2-F3-divergence-2014 (UST_10Y + EUR + JPY, two panels with different holiday
calendars) not reproduce live m0b (1.7440)? Test target-date hypotheses for mixed-calendar cards."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import t2_core as C
import build_truth as BT

LIVE = json.loads((C.HERE / "live_scores.json").read_text())["m0b"]["cards"]
names = sys.argv[1:] or ["t2-F3-divergence-2014", "t2-F3-dollar-vortex-2022"]
sib = BT.sibling_table()


def long_series(name, a):
    stem = BT.panel_stem(name, a)
    src, _ = BT.source(a)
    for end, unit, path in sorted(sib[(stem, a)], reverse=True):
        df = pd.read_parquet(path)
        g = df[df["asset"].astype(str) == a]
        return pd.Series(g["value"].astype(float).to_numpy(), index=pd.to_datetime(g["date"]).to_numpy()).sort_index()


for name in names:
    card = C.card_of(name)
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    base, cells = C.m0x_draws(name)
    S = C.m0b_draws(name, cells)
    assets = [str(a) for a in card["targets"]["asset_ids"]]
    ser = {a: long_series(name, a) for a in assets}
    fut = {a: ser[a][ser[a].index > asof] for a in assets}
    print(name, "asof", asof.date(), "live", LIVE[name])
    for h in sorted(int(h) for h in card["targets"]["horizons"]):
        print("  h", h, {a: str(fut[a].index[h - 1].date()) for a in assets})
    inter = None
    union = None
    for a in assets:
        inter = fut[a].index if inter is None else inter.intersection(fut[a].index)
        union = fut[a].index if union is None else union.union(fut[a].index)

    def val(a, d):
        s = ser[a][ser[a].index <= d]
        return float(s.iloc[-1])

    hyps = {
        "own h-th row": lambda a, h: float(fut[a].iloc[h - 1]),
        "h-th date of intersection calendar": lambda a, h: val(a, inter[h - 1]),
        "h-th date of union calendar": lambda a, h: val(a, union[h - 1]),
        "asof + h BDay": lambda a, h: val(a, asof + pd.offsets.BDay(h)),
    }
    for a0 in assets:
        hyps[f"h-th row date of {a0}"] = lambda a, h, a0=a0: val(a, fut[a0].index[h - 1])
    hyps["h-th row of intersection frame (row h-1 of joined rows)"] = lambda a, h: float(ser[a][inter[h - 1]])
    for label, f in hyps.items():
        y = np.array([f(a, h) for a, h in cells])
        print(f"   {label:<58} {min(4.0, C.card_score(card, base, S, y)):.4f}")
