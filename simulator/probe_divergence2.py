"""Brute-force the target dates of t2-F3-divergence-2014 against live m0b (1.7440)."""
import itertools
import json

import numpy as np
import pandas as pd

import t2_core as C
import build_truth as BT
import probe_divergence as P  # noqa: F401  (reuses long_series; prints its own table first)

name = "t2-F3-divergence-2014"
LIVE = json.loads((C.HERE / "live_scores.json").read_text())["m0b"]["cards"][name]
card = C.card_of(name)
asof = pd.Timestamp(card["provenance"]["data_cutoff"])
base, cells = C.m0x_draws(name)
S = C.m0b_draws(name, cells)
assets = [str(a) for a in card["targets"]["asset_ids"]]
ser = {a: P.long_series(name, a) for a in assets}
for a in assets:
    s = ser[a]
    print(a, "rows 2014-12-22..2015-01-08:", [str(d.date()) for d in s["2014-12-22":"2015-01-08"].index])
    print(a, "rows 2015-03-30..2015-04-08:", [str(d.date()) for d in s["2015-03-30":"2015-04-08"].index])


def val(a, d):
    s = ser[a][ser[a].index <= d]
    return float(s.iloc[-1])


d63 = pd.date_range("2014-12-22", "2015-01-09", freq="D")
d126 = pd.date_range("2015-03-25", "2015-04-10", freq="D")
hits = []
for a_dates in itertools.product(d63, d126):
    y = np.array([val(a, a_dates[0] if h == 63 else a_dates[1]) for a, h in cells])
    sc = C.card_score(card, base, S, y)
    if abs(sc - LIVE) < 6e-5:
        hits.append((str(a_dates[0].date()), str(a_dates[1].date()), sc))
print("common-date hits:", hits)
# per-panel dates: rates date and fx date independent, per horizon
hits2 = []
r63 = pd.date_range("2014-12-29", "2015-01-07", freq="B")
r126 = pd.date_range("2015-03-31", "2015-04-08", freq="B")
for dr1, df1, dr2, df2 in itertools.product(r63, r63, r126, r126):
    y = []
    for a, h in cells:
        d = (dr1 if a.startswith("UST") else df1) if h == 63 else (dr2 if a.startswith("UST") else df2)
        y.append(val(a, d))
    sc = C.card_score(card, base, S, np.array(y))
    if abs(sc - LIVE) < 6e-5:
        hits2.append((str(dr1.date()), str(df1.date()), str(dr2.date()), str(df2.date()), round(sc, 5)))
print("per-panel hits (rates63, fx63, rates126, fx126):", hits2)
