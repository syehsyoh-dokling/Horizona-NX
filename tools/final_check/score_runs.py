"""Score a directory of container outputs (out_<tag>/<unit>/forecast.parquet) on the 22 -ps1 cards
and compare each forecast matrix with the local engine's draws for a given rule.

usage: score_runs.py <inputs_dir> <answers_dir> <out_dir> <rule> <out_json>
  rule: newscale | m0 | <variant spec like "widen=0.8,drift=zero"> (uniform across families)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import rawscore as R
from forecast_core import m0_exact as M

INPUTS, ANSWERS, OUTDIR = (pathlib.Path(p) for p in sys.argv[1:4])
RULE = sys.argv[4]
OUT = pathlib.Path(sys.argv[5])
if len(sys.argv) > 6:  # optional: compare against another engine copy (e.g. sim/agent_fixed)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "m0_exact_alt", pathlib.Path(sys.argv[6]) / "forecast_core" / "m0_exact.py")
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)
    print("local engine:", sys.argv[6])

NEWSCALE = {
    "F1": {"t_nu": 5.0, "drift": "zero"},
    "F2": {"widen": 1.5, "drift": "zero"},
    "F3": {"widen": 1.25, "t_nu": 5.0, "drift": "zero"},
    "F4": {"widen": 3.0},
}


def parse_spec(spec: str) -> dict:
    v = {}
    for part in spec.split(","):
        k, _, val = part.partition("=")
        if not k:
            continue
        try:
            v[k] = [float(x) for x in val.split(":")] if ":" in val else float(val)
        except ValueError:
            v[k] = val
    return v


rows = []
for unit in sorted(INPUTS.glob("*-ps1")):
    card = R.read_card(unit)
    uid = card["task"]["id"]
    fam = uid.split("-")[1]
    cells = R.cells_of(card)
    y = R.realized_vector(ANSWERS / f"{uid}.parquet", cells)
    fp = OUTDIR / uid / "forecast.parquet"
    if not fp.is_file():
        rows.append({"unit": uid, "error": "no forecast.parquet"})
        continue
    S = R.matrix_from_parquet(fp, cells)
    comp = R.composite(S, y, card)
    mean, cov, _ = R.m0_mean_cov(card, R.panels_dir_of(unit))
    div = R.divisors(mean, cov)
    sc = R.scaled(comp, div)
    if RULE == "newscale":
        variant = dict(NEWSCALE[fam])
    elif RULE.startswith("newscale+"):          # e.g. newscale+widen_gate=5 (the image's T2_M0_VARIANT)
        variant = {**dict(NEWSCALE[fam]), **parse_spec(RULE.split("+", 1)[1])}
    elif RULE == "m0":
        variant = None
    else:
        variant = parse_spec(RULE)
    S_loc, _, _ = M.draws(card, R.panels_dir_of(unit), variant=variant or None)
    maxdiff = float(np.max(np.abs(S - S_loc))) if S.shape == S_loc.shape else float("nan")
    stdout = (OUTDIR / uid / "stdout.json").read_text(encoding="utf-8", errors="replace").strip()
    rows.append({"unit": uid, "family": fam, "cells": len(cells), "raw": comp["composite"],
                 "scaled": sc, "clipped": min(sc, R.CLIP), "n_draws": int(S.shape[0]),
                 "max_abs_diff_vs_local": maxdiff, "stdout": stdout[-300:]})

ok = [r for r in rows if "error" not in r]
print(f"{'unit':46s} fam cells   raw        scaled   clip   maxdiff_vs_local")
for r in rows:
    if "error" in r:
        print(f"{r['unit']:46s} {r['error']}")
        continue
    print(f"{r['unit']:46s} {r['family']}  {r['cells']:2d}   {r['raw']:.6f}  {r['scaled']:.4f}  {r['clipped']:.4f}  {r['max_abs_diff_vs_local']:.2e}")
clipped = np.array([r["clipped"] for r in ok])
fams = {}
for r in ok:
    fams.setdefault(r["family"], []).append(r["clipped"])
print(f"\nmean clipped scaled {clipped.mean():.4f}  odd {clipped[0::2].mean():.4f} even {clipped[1::2].mean():.4f}  "
      + "  ".join(f"{f} {np.mean(v):.3f}" for f, v in sorted(fams.items())))
print(f"max |container - local| over all cards: {max(r['max_abs_diff_vs_local'] for r in ok):.3e}")
json.dump(rows, OUT.open("w"), indent=1)
