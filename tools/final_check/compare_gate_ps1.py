"""Horizon-gated widening on the 22 -ps1 cards (fixed engine + widen_gate rule in sim/agent_gate).
Compares: newscalefix (reference), newscalegate at G = 5 / 10 / 20 BD (newscale rules, widening
undone for cells with declared horizon <= G), exact M0, plain RW. Also verifies that the gate engine
reproduces the fixed engine exactly when the gate is off.
usage: compare_gate_ps1.py <inputs_dir> <answers_dir> <out_json>
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

import rawscore as R

HERE = pathlib.Path(__file__).resolve().parent
INPUTS, ANSWERS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])


def load_engine(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIX = load_engine(HERE / "agent_fixed" / "forecast_core" / "m0_exact.py", "m0_fixed")
GATE = load_engine(HERE / "agent_gate" / "forecast_core" / "m0_exact.py", "m0_gate")

NEWSCALE = {
    "F1": {"t_nu": 5.0, "drift": "zero"},
    "F2": {"widen": 1.5, "drift": "zero"},
    "F3": {"widen": 1.25, "t_nu": 5.0, "drift": "zero"},
    "F4": {"widen": 3.0},
}


def ns(fam, gate=None):
    v = dict(NEWSCALE[fam])
    if gate is not None:
        v["widen_gate"] = float(gate)   # same string the image gets via T2_M0_VARIANT="widen_gate=G"
    return v


RULES = {
    "newscalefix": (FIX, lambda f: ns(f)),
    "gate5": (GATE, lambda f: ns(f, 5)),
    "gate10": (GATE, lambda f: ns(f, 10)),
    "gate20": (GATE, lambda f: ns(f, 20)),
    "m0": (FIX, lambda f: None),
    "rw": (FIX, lambda f: {"drift": "zero"}),
    "_gate_off_check": (GATE, lambda f: ns(f)),
}

cards = []
for unit in sorted(INPUTS.glob("*-ps1")):
    card = R.read_card(unit)
    uid = card["task"]["id"]
    fam = uid.split("-")[1]
    cells = R.cells_of(card)
    y = R.realized_vector(ANSWERS / f"{uid}.parquet", cells)
    pdir = R.panels_dir_of(unit)
    mean, cov, _ = R.m0_mean_cov(card, pdir)
    div = R.divisors(mean, cov)
    panel = card["panels"]["panel_ids"][0]
    ptype = "rates" if panel == "rates_daily" else ("em_fx" if "em_fx" in panel else "g10_fx")
    row = {"unit": uid, "family": fam, "ptype": ptype, "horizons": sorted({h for _, h in cells}), "scores": {}, "raw": {}}
    draws = {}
    for name, (eng, rule) in RULES.items():
        S, c2, _ = eng.draws(card, pdir, variant=rule(fam) or None)
        assert c2 == cells
        draws[name] = S
        comp = R.composite(S, y, card)
        row["scores"][name] = min(R.scaled(comp, div), R.CLIP)
        row["raw"][name] = comp["composite"]
    row["gate_off_maxdiff"] = float(np.max(np.abs(draws["_gate_off_check"] - draws["newscalefix"])))
    row["gate5_vs_10_maxdiff"] = float(np.max(np.abs(draws["gate5"] - draws["gate10"])))
    cards.append(row)

names = [n for n in RULES if not n.startswith("_")]
print(f"{'unit':44s} fam ptype  hor    " + " ".join(f"{n:>12s}" for n in names))
for r in cards:
    print(f"{r['unit']:44s} {r['family']} {r['ptype']:6s} {str(r['horizons']):7s}" + " ".join(f"{r['scores'][n]:12.4f}" for n in names))
mat = np.array([[r["scores"][n] for n in names] for r in cards])
print(f"{'MEAN (clip 8)':65s}" + " ".join(f"{v:12.4f}" for v in mat.mean(axis=0)))
print(f"{'odd half':65s}" + " ".join(f"{v:12.4f}" for v in mat[0::2].mean(axis=0)))
print(f"{'even half':65s}" + " ".join(f"{v:12.4f}" for v in mat[1::2].mean(axis=0)))
groups = {}
for key in ("family", "ptype"):
    for g in sorted({r[key] for r in cards}):
        idx = [i for i, r in enumerate(cards) if r[key] == g]
        groups[f"{key}={g}"] = {n: float(v) for n, v in zip(names, mat[idx].mean(axis=0))}
        print(f"{key + '=' + g + f' (n={len(idx)})':65s}" + " ".join(f"{v:12.4f}" for v in mat[idx].mean(axis=0)))
print(f"\ngate engine with gate off == fixed engine: max |diff| {max(r['gate_off_maxdiff'] for r in cards):.2e}")
print(f"gate5 vs gate10 draws on the practice set: max |diff| {max(r['gate5_vs_10_maxdiff'] for r in cards):.2e} (all horizons <= 4)")

rng = np.random.default_rng(0)
boot_idx = rng.integers(0, len(cards), size=(20000, len(cards)))
summary = {}
for ref in ("newscalefix", "m0"):
    j_ref = names.index(ref)
    print(f"\nvs {ref}:")
    summary[ref] = {}
    for j, n in enumerate(names):
        if n == ref:
            continue
        d = mat[:, j] - mat[:, j_ref]
        loo = [np.delete(d, i).mean() for i in range(len(d))]
        boot = d[boot_idx].mean(axis=1)
        s = {"mean_diff": float(d.mean()), "odd_diff": float(d[0::2].mean()), "even_diff": float(d[1::2].mean()),
             "better": int((d < -1e-9).sum()), "worse": int((d > 1e-9).sum()), "loo_min": float(min(loo)), "loo_max": float(max(loo)),
             "p_boot_better": float((boot < 0).mean()), "max_gain": float(d.min()), "max_loss": float(d.max())}
        summary[ref][n] = s
        print(f"  {n:12s} diff {s['mean_diff']:+.4f} (odd {s['odd_diff']:+.4f} even {s['even_diff']:+.4f})  better {s['better']:2d} worse {s['worse']:2d}  "
              f"LOO [{s['loo_min']:+.4f},{s['loo_max']:+.4f}]  P(better) {s['p_boot_better']:.2f}  max gain {s['max_gain']:+.3f} loss {s['max_loss']:+.3f}")
json.dump({"cards": cards, "names": names, "means": {n: float(v) for n, v in zip(names, mat.mean(axis=0))},
           "odd": {n: float(v) for n, v in zip(names, mat[0::2].mean(axis=0))}, "even": {n: float(v) for n, v in zip(names, mat[1::2].mean(axis=0))},
           "groups": groups, "summary": summary}, OUT.open("w"), indent=1)
print("wrote", OUT)
