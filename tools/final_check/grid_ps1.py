"""Score exact M0, the submitted newscale configuration and generic short-horizon variants on the
22 -ps1 cards with the rebuilt scorer (rawscore.py). Analysis only; nothing here ships.

Every variant is a rule of the card's FAMILY, HORIZON RANGE and PANEL TYPE only - never of the
card id, never of its outcome. Output: per-card table, per-family / per-panel means, odd-even
split halves, leave-one-out stability of the best rules, and width calibration diagnostics.

usage: grid_ps1.py <inputs_dir> <answers_dir> <out_json>
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import rawscore as R
from forecast_core import m0_exact as M

INPUTS = pathlib.Path(sys.argv[1])
ANSWERS = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])

NEWSCALE = {  # image/Dockerfile.newscale T2_M0_FAMILY_VARIANT
    "F1": {"t_nu": 5.0, "drift": "zero"},
    "F2": {"widen": 1.5, "drift": "zero"},
    "F3": {"widen": 1.25, "t_nu": 5.0, "drift": "zero"},
    "F4": {"widen": 3.0},
}


class Card:
    def __init__(self, unit_dir: pathlib.Path):
        self.dir = unit_dir
        self.card = R.read_card(unit_dir)
        self.id = self.card["task"]["id"]
        self.family = self.id.split("-")[1]
        self.panel = self.card["panels"]["panel_ids"][0]
        self.ptype = "rates" if self.panel == "rates_daily" else ("em_fx" if "em_fx" in self.panel else "g10_fx")
        self.cells = R.cells_of(self.card)
        self.hmax = max(h for _, h in self.cells)
        self.y = R.realized_vector(ANSWERS / f"{self.id}.parquet", self.cells)
        self.panels = R.panels_dir_of(unit_dir)
        mean, cov, _ = R.m0_mean_cov(self.card, self.panels)
        self.m0_mean, self.m0_cov = mean, cov
        self.div = R.divisors(mean, cov)

    def draws(self, variant: dict | None):
        s, cells, _info = M.draws(self.card, self.panels, variant=variant or None)
        assert cells == self.cells
        return s

    def score(self, variant: dict | None) -> dict:
        S = self.draws(variant)
        comp = R.composite(S, self.y, self.card)
        sc = R.scaled(comp, self.div)
        return {"raw": comp["composite"], "scaled": sc, "clipped": min(sc, R.CLIP),
                "marg_n": comp["marginal"] / self.div[0], "joint_n": comp["joint"] / self.div[1],
                "tail_n": comp["tail"] / self.div[2]}


cards = [Card(u) for u in sorted(INPUTS.glob("*-ps1"))]
print(f"{len(cards)} cards")


def rule_newscale(c: Card):
    return dict(NEWSCALE[c.family])


def rule_m0(c: Card):
    return None


def make_rule(widen=None, t_nu=None, drift=None, ewma=None, keep_family=False):
    """Generic short-horizon rule (applies to every card with hmax <= 5 BD, i.e. all 22)."""
    def rule(c: Card):
        v = dict(NEWSCALE[c.family]) if keep_family else {}
        if widen is not None:
            v["widen"] = float(widen) * (float(v.get("widen", 1.0)) if keep_family else 1.0)
        if t_nu is not None:
            v["t_nu"] = float(t_nu)
        if drift is not None:
            v["drift"] = drift
        if ewma is not None:
            v["ewma"] = list(ewma)
        return v or None
    return rule


def evaluate(rule) -> dict:
    per = {c.id: c.score(rule(c)) for c in cards}
    clipped = np.array([per[c.id]["clipped"] for c in cards])
    out = {"per_card": per, "mean": float(clipped.mean()),
           "odd": float(clipped[0::2].mean()), "even": float(clipped[1::2].mean())}
    for key in ("family", "ptype"):
        groups = {}
        for i, c in enumerate(cards):
            groups.setdefault(getattr(c, key), []).append(clipped[i])
        out[key] = {k: float(np.mean(v)) for k, v in sorted(groups.items())}
    return out


results = {}
results["m0"] = evaluate(rule_m0)
results["newscale"] = evaluate(rule_newscale)
print(f"exact M0   mean {results['m0']['mean']:.4f}  odd {results['m0']['odd']:.4f} even {results['m0']['even']:.4f}  fam {results['m0']['family']}  ptype {results['m0']['ptype']}")
print(f"newscale   mean {results['newscale']['mean']:.4f}  odd {results['newscale']['odd']:.4f} even {results['newscale']['even']:.4f}  fam {results['newscale']['family']}  ptype {results['newscale']['ptype']}")

# ---- grid of generic short-horizon rules (uniform across families; family widening dropped)
grid = []
for widen, t_nu, drift, ewma in itertools.product(
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5],
        [None, 5.0, 8.0],
        [None, "zero"],
        [None, (0.94, 1.0), (0.94, 0.5), (0.97, 1.0), (0.97, 0.5)]):
    name = f"w={widen},t={t_nu},d={drift},ewma={ewma}"
    res = evaluate(make_rule(widen, t_nu, drift, ewma))
    res_light = {k: v for k, v in res.items() if k != "per_card"}
    grid.append((name, res_light))
grid.sort(key=lambda kv: kv[1]["mean"])
print("\nTop 15 generic rules (uniform; mean clipped scaled score):")
for name, r in grid[:15]:
    print(f"  {r['mean']:.4f}  odd {r['odd']:.4f} even {r['even']:.4f}  {name}  fam {r['family']}  ptype {r['ptype']}")

# ---- family widening kept, uniform short-horizon multiplier on top (tests "is the family width right")
grid_k = []
for widen in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    name = f"newscale x widen {widen}"
    res = evaluate(make_rule(widen=widen, keep_family=True))
    grid_k.append((name, {k: v for k, v in res.items() if k != "per_card"}))
print("\nNewscale family variants kept, width multiplied:")
for name, r in grid_k:
    print(f"  {r['mean']:.4f}  odd {r['odd']:.4f} even {r['even']:.4f}  {name}  fam {r['family']}")

# ---- per-family width scan on plain M0 (drift zero): which widen minimises each family?
print("\nPer-family width scan (M0, drift zero, normal innovations):")
fam_scan = {}
for widen in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
    res = evaluate(make_rule(widen=widen, drift="zero"))
    fam_scan[widen] = res["family"]
    print(f"  widen {widen:<5}  " + "  ".join(f"{f} {v:.3f}" for f, v in res["family"].items()) + f"   all {res['mean']:.4f}")

# ---- calibration diagnostics: PIT / |z| of the realised value under M0's normal and newscale's draws
diag = []
for c in cards:
    sd = np.sqrt(np.diag(c.m0_cov))
    z = (c.y - c.m0_mean) / sd
    S_ns = c.draws(rule_newscale(c))
    pit_ns = (S_ns < c.y).mean(axis=0)
    for j, (a, h) in enumerate(c.cells):
        diag.append({"unit": c.id, "family": c.family, "ptype": c.ptype, "asset": a, "h": h,
                     "z_m0": float(z[j]), "pit_m0": float(norm.cdf(z[j])), "pit_newscale": float(pit_ns[j])})
dz = pd.DataFrame(diag)
print("\nCalibration of M0's width (z = realised - mean, in M0 sd units): n cells", len(dz))
print(f"  mean|z| {dz.z_m0.abs().mean():.3f} (calibrated normal: 0.798)   rms z {np.sqrt((dz.z_m0**2).mean()):.3f} (1.0)"
      f"   inside 50% {(dz.z_m0.abs() < 0.6745).mean():.2f}   inside 90% {(dz.z_m0.abs() < 1.645).mean():.2f}")
for key in ("family", "ptype"):
    for g, d in dz.groupby(key):
        print(f"  {key}={g:6s} n={len(d):2d} mean|z| {d.z_m0.abs().mean():.3f} rms {np.sqrt((d.z_m0**2).mean()):.3f}"
              f"  in50 {(d.z_m0.abs() < 0.6745).mean():.2f} in90 {(d.z_m0.abs() < 1.645).mean():.2f}"
              f"  PIT newscale in[.25,.75] {((d.pit_newscale > .25) & (d.pit_newscale < .75)).mean():.2f}")

json.dump({"m0": results["m0"], "newscale": results["newscale"], "grid": grid, "grid_keep_family": grid_k,
           "family_width_scan": {str(k): v for k, v in fam_scan.items()}, "diag": diag,
           "cards": [{"id": c.id, "family": c.family, "ptype": c.ptype, "cells": c.cells, "div": c.div} for c in cards]},
          OUT.open("w"), indent=1)
print("\nwrote", OUT)
