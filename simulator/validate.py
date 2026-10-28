"""Validate the simulator against LIVE per-card leaderboard scores (live_scores.json).

1. m0b (vendored reference CLI walk, F1/F3 scale from the last 504 changes; live 1.1368):
   local per-card score under truth/scorer hypotheses vs the live per-card score.
2. m0x (exact M0; live 1.0000 with two cards at 0.9985 / 0.9994): local is 1 by construction;
   test whether a different M0 cell order explains the two live exceptions.
Writes validation.json and, into truth.json, per card: live_m0b, local_m0b, live_match.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import zlib

import numpy as np

import t2_core as C

HERE = pathlib.Path(__file__).resolve().parent
LIVE = json.loads((HERE / "live_scores.json").read_text())
truth = C.load_truth()
live_m0b = LIVE["m0b"]["cards"]
live_m0x = LIVE["m0x exact"]["cards"]


def y_alt(name, cells, key):
    by = {(c["asset"], c["horizon"]): c for c in truth[name]["cells"]}
    out = []
    for a, h in cells:
        c = by[(a, h)]
        v = c["value"] if key == "value" else (c.get("alt") or {}).get(key)
        if v is None:
            v = c["value"]
        out.append(v)
    return np.array(out, dtype=float)


rows = []
hyp = {}
for name in sorted(live_m0b):
    base, cells = C.m0x_draws(name)
    S = C.m0b_draws(name, cells)
    card = C.card_of(name)
    r = {"unit": name, "live": live_m0b[name], "n_cells": len(cells),
         "verified": truth[name]["verified"], "ttype": truth[name]["target_type"]}
    for key in ("value", "logsum", "bday_rule"):
        y = y_alt(name, cells, key)
        r[key] = min(C.CLIP, C.card_score(card, base, S, y))
    y = y_alt(name, cells, "value")
    r["cov_tail"] = min(C.CLIP, C.card_score(card, base, S, y, tail="coverage"))
    r["comp_ratio"] = min(C.CLIP, C.card_score(card, base, S, y, mode="composite_ratio"))
    rows.append(r)


def summary(key, sel=None):
    rr = [r for r in rows if sel is None or sel(r)]
    e = np.array([abs(r[key] - r["live"]) for r in rr])
    lv = np.array([r["live"] for r in rr])
    lc = np.array([r[key] for r in rr])
    return {"n": len(rr), "exact(<0.0002)": int(np.sum(e < 2e-4)), "within0.002": int(np.sum(e < 2e-3)),
            "within0.01": int(np.sum(e < 1e-2)), "median_abs_err": float(np.median(e)),
            "max_abs_err": float(e.max()), "corr": float(np.corrcoef(lv, lc)[0, 1]),
            "mean_live": float(lv.mean()), "mean_local": float(lc.mean())}


report = {"m0b": {}}
print("1. m0b: local per-card score vs live (71 cards; live primary 1.1368)")
for key, label in (("value", "h-th row, sum of returns, per-component, pinball  [ADOPTED]"),
                   ("logsum", "  log_return target as sum(log(1+r))"),
                   ("bday_rule", "  target = as-of + h business days"),
                   ("cov_tail", "  coverage tail instead of pinball"),
                   ("comp_ratio", "  composite ratio instead of per-component")):
    s = summary(key)
    report["m0b"][key] = s
    print(f"  {label:<62} primary {s['mean_local']:.4f}  exact {s['exact(<0.0002)']:2d}/71  "
          f"<0.002 {s['within0.002']:2d}  <0.01 {s['within0.01']:2d}  median|err| {s['median_abs_err']:.5f}  "
          f"max {s['max_abs_err']:.4f}  corr {s['corr']:.5f}")
s = summary("logsum", lambda r: r["ttype"] == "log_return")
s2 = summary("value", lambda r: r["ttype"] == "log_return")
print(f"  log_return cards only ({s['n']}): sum(r) exact {s2['exact(<0.0002)']}, sum(log1p r) exact {s['exact(<0.0002)']}")
fam = {}
for r in rows:
    fam.setdefault(C.family(r["unit"]), []).append((r["live"], r["value"]))
print("  family means live/local: " + "  ".join(
    f"{f} {np.mean([a for a, _ in v]):.4f}/{np.mean([b for _, b in v]):.4f}" for f, v in sorted(fam.items())))
report["m0b"]["family"] = {f: [float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))] for f, v in fam.items()}
print("  cards off by >= 0.0002:")
for r in sorted(rows, key=lambda r: -abs(r["value"] - r["live"])):
    if abs(r["value"] - r["live"]) >= 2e-4:
        print(f"    {r['unit']:<38} live {r['live']:.4f} local {r['value']:.4f}  (logsum {r['logsum']:.4f}, "
              f"bday {r['bday_rule']:.4f})  verified={r['verified']}")

# ---- 2. exact M0 live exceptions
print("\n2. exact M0 live: all 1.0000 except", {k: v for k, v in live_m0x.items() if v != 1.0})
report["m0x"] = {}
for name in [k for k, v in live_m0x.items() if v != 1.0]:
    card = C.card_of(name)
    ours, cells = C.m0x_draws(name)             # our upload = sorted (asset, horizon) order
    y = y_alt(name, cells, "value")
    assets_card = [str(a) for a in card["targets"]["asset_ids"]]
    hs_card = [int(h) for h in card["targets"]["horizons"]]
    # M0 with Z columns assigned in another cell order: same mean/cov, different column<->cell map
    mean, L = None, None
    orders = {"card list order (asset, horizon)": [(a, h) for a in assets_card for h in hs_card],
              "card list order (horizon, asset)": [(a, h) for h in hs_card for a in assets_card],
              "sorted (horizon, asset)": [(a, h) for h in sorted(hs_card) for a in sorted(assets_card)]}
    seed = zlib.crc32(name.encode()) & 0x7FFFFFFF
    mu_s = ours.mean(axis=0)
    res = {}
    for label, order in orders.items():
        # rebuild M0 in that order from the exact mean/cov implied by m0_exact (recompute analytically)
        import pandas as pd
        asof = pd.Timestamp(card["provenance"]["data_cutoff"])
        ttype = str(card["targets"].get("target_type", "level"))
        A = sorted(assets_card)
        hist = {a: C.M.history(C.UNITS / name, a, asof) for a in A}
        frame = pd.concat({a: C.M.steps(hist[a], ttype) for a in A}, axis=1, join="inner").dropna()
        X = frame[A].to_numpy(float)
        mu, Sigma = X.mean(axis=0), np.atleast_2d(np.cov(X, rowvar=False))
        ai = {a: i for i, a in enumerate(A)}
        anchor = {a: (0.0 if ttype == "log_return" else float(hist[a].iloc[-1])) for a in A}
        d = len(order)
        mean = np.array([anchor[a] + h * mu[ai[a]] for a, h in order])
        cov = np.array([[min(h, g) * Sigma[ai[a], ai[b]] for (b, g) in order] for (a, h) in order])
        cov = cov + 1e-10 * np.eye(d)
        Lc = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
        Z = np.random.default_rng(seed).standard_normal((500, d))
        base_o = mean + Z @ Lc.T
        col = {c: j for j, c in enumerate(order)}
        base = np.column_stack([base_o[:, col[c]] for c in cells])
        res[label] = C.card_score(card, base, ours, y)
    report["m0x"][name] = {"live": live_m0x[name], "orders": res}
    print(f"  {name}: live {live_m0x[name]:.4f}; our draws scored against M0 built in another cell order:")
    for label, v in res.items():
        print(f"      {label:<36} {v:.4f}")

# ---- write back
for r in rows:
    t = truth[r["unit"]]
    t["live_m0b"] = r["live"]
    t["local_m0b"] = round(r["value"], 6)
    t["live_match"] = bool(abs(r["value"] - r["live"]) < 2e-3)
(HERE / "truth.json").write_text(json.dumps(truth, indent=1))
(HERE / "validation.json").write_text(json.dumps({"report": report, "rows": rows}, indent=1))
print("\nlive_match (|local-live| < 0.002):", sum(truth[r['unit']]['live_match'] for r in rows), "/ 71;",
      "data-verified:", sum(truth[r['unit']]['verified'] for r in rows), "/ 71")
