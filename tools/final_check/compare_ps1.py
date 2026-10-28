"""Per-card comparison of named rules on the 22 -ps1 cards + leave-one-out and card-bootstrap
of the mean difference against newscale. Analysis only.

usage: compare_ps1.py <inputs_dir> <answers_dir> <out_json>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import rawscore as R
from forecast_core import m0_exact as M

INPUTS, ANSWERS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])

NEWSCALE = {
    "F1": {"t_nu": 5.0, "drift": "zero"},
    "F2": {"widen": 1.5, "drift": "zero"},
    "F3": {"widen": 1.25, "t_nu": 5.0, "drift": "zero"},
    "F4": {"widen": 3.0},
}


def rule_newscale(fam, hmax):
    return dict(NEWSCALE[fam])


def rule_m0(fam, hmax):
    return None


def rule_rw(fam, hmax):            # plain random walk: M0 width, zero drift, normal
    return {"drift": "zero"}


def rule_short_rw(fam, hmax):      # generic by horizon: short cards -> plain RW; long cards -> newscale
    return {"drift": "zero"} if hmax <= 5 else dict(NEWSCALE[fam])


def rule_ns_x07(fam, hmax):        # newscale family variants, widths x0.7 (fitted multiplier; for the record)
    v = dict(NEWSCALE[fam])
    v["widen"] = 0.7 * float(v.get("widen", 1.0))
    return v


def rule_rw_w08(fam, hmax):
    return {"drift": "zero", "widen": 0.8}


def rule_rw_t8(fam, hmax):
    return {"drift": "zero", "t_nu": 8.0}


RULES = {"m0": rule_m0, "newscale": rule_newscale, "rw": rule_rw, "short_rw": rule_short_rw,
         "rw_w0.8": rule_rw_w08, "rw_t8": rule_rw_t8, "ns_x0.7": rule_ns_x07}

cards = []
for unit in sorted(INPUTS.glob("*-ps1")):
    card = R.read_card(unit)
    uid = card["task"]["id"]
    fam = uid.split("-")[1]
    cells = R.cells_of(card)
    hmax = max(h for _, h in cells)
    y = R.realized_vector(ANSWERS / f"{uid}.parquet", cells)
    pdir = R.panels_dir_of(unit)
    mean, cov, _ = R.m0_mean_cov(card, pdir)
    div = R.divisors(mean, cov)
    panel = card["panels"]["panel_ids"][0]
    ptype = "rates" if panel == "rates_daily" else ("em_fx" if "em_fx" in panel else "g10_fx")
    row = {"unit": uid, "family": fam, "ptype": ptype, "cells": len(cells), "hmax": hmax, "scores": {}, "raw": {}}
    for name, rule in RULES.items():
        S, _, _ = M.draws(card, pdir, variant=rule(fam, hmax) or None)
        comp = R.composite(S, y, card)
        sc = R.scaled(comp, div)
        row["scores"][name] = min(sc, R.CLIP)
        row["raw"][name] = comp["composite"]
    cards.append(row)

names = list(RULES)
print(f"{'unit':46s} fam ptype  n " + " ".join(f"{n:>9s}" for n in names))
for r in cards:
    print(f"{r['unit']:46s} {r['family']} {r['ptype']:6s} {r['cells']:2d} " + " ".join(f"{r['scores'][n]:9.4f}" for n in names))
mat = np.array([[r["scores"][n] for n in names] for r in cards])
print(f"{'MEAN':60s} " + " ".join(f"{v:9.4f}" for v in mat.mean(axis=0)))
print(f"{'odd half':60s} " + " ".join(f"{v:9.4f}" for v in mat[0::2].mean(axis=0)))
print(f"{'even half':60s} " + " ".join(f"{v:9.4f}" for v in mat[1::2].mean(axis=0)))
print(f"{'MEDIAN':60s} " + " ".join(f"{v:9.4f}" for v in np.median(mat, axis=0)))
for key in ("family", "ptype"):
    for g in sorted({r[key] for r in cards}):
        idx = [i for i, r in enumerate(cards) if r[key] == g]
        print(f"{key + '=' + g + f' (n={len(idx)})':60s} " + " ".join(f"{v:9.4f}" for v in mat[idx].mean(axis=0)))

j_ns = names.index("newscale")
print("\nvs newscale: mean diff, cards better/worse, LOO range of mean diff, bootstrap P(diff<0)")
rng = np.random.default_rng(0)
boot_idx = rng.integers(0, len(cards), size=(20000, len(cards)))
summary = {}
for j, n in enumerate(names):
    if n == "newscale":
        continue
    d = mat[:, j] - mat[:, j_ns]
    loo = [np.delete(d, i).mean() for i in range(len(d))]
    boot = d[boot_idx].mean(axis=1)
    summary[n] = {"mean_diff": float(d.mean()), "better": int((d < -1e-9).sum()), "worse": int((d > 1e-9).sum()),
                  "loo_min": float(min(loo)), "loo_max": float(max(loo)), "p_boot_better": float((boot < 0).mean()),
                  "largest_single_card_gain": float(d.min()), "largest_single_card_loss": float(d.max())}
    print(f"  {n:10s} diff {d.mean():+.4f}  better {summary[n]['better']:2d} worse {summary[n]['worse']:2d}  "
          f"LOO [{min(loo):+.4f},{max(loo):+.4f}]  P(better) {summary[n]['p_boot_better']:.2f}  "
          f"max single gain {d.min():+.3f} loss {d.max():+.3f}")

json.dump({"cards": cards, "names": names, "summary_vs_newscale": summary}, OUT.open("w"), indent=1)
