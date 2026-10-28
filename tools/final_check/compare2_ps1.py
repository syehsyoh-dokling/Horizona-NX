"""Per-card comparison on the 22 -ps1 cards with the CORRECTED divisor (declared horizon, M0-BASELINE
section 3.7 strict rule) and two engines:
  asis  = repo HEAD forecast_core/m0_exact.py (byte-identical to the newscale image): ratio >= 2
  fixed = sim/agent_fixed/forecast_core/m0_exact.py: ratio > 2 (the specification)
Rules are functions of family / horizon range only. Analysis only; nothing here ships.

usage: compare2_ps1.py <inputs_dir> <answers_dir> <out_json>
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import rawscore as R

HERE = pathlib.Path(__file__).resolve().parent
INPUTS, ANSWERS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])


def load_engine(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ENG = {"asis": load_engine(HERE.parent / "repo" / "agent" / "forecast_core" / "m0_exact.py", "m0_asis"),
       "fixed": load_engine(HERE / "agent_fixed" / "forecast_core" / "m0_exact.py", "m0_fixed")}

NEWSCALE = {
    "F1": {"t_nu": 5.0, "drift": "zero"},
    "F2": {"widen": 1.5, "drift": "zero"},
    "F3": {"widen": 1.25, "t_nu": 5.0, "drift": "zero"},
    "F4": {"widen": 3.0},
}


def r_m0(fam):
    return None


def r_newscale(fam):
    return dict(NEWSCALE[fam])


def r_rw(fam):                 # plain random walk: M0 width, zero drift, normal innovations
    return {"drift": "zero"}


def r_rw_t8(fam):
    return {"drift": "zero", "t_nu": 8.0}


def r_rw_t5(fam):
    return {"drift": "zero", "t_nu": 5.0}


def r_ns_x07(fam):             # fitted multiplier on newscale's widths (for the record only)
    v = dict(NEWSCALE[fam])
    v["widen"] = 0.7 * float(v.get("widen", 1.0))
    return v


def r_widen(w):
    def rule(fam):
        return {"drift": "zero", "widen": float(w)}
    return rule


RULES = {  # (engine, rule)
    "newscale_asis": ("asis", r_newscale),     # the current Final entry, as the image runs
    "m0_asis": ("asis", r_m0),
    "m0_fixed": ("fixed", r_m0),               # exact M0 = the organizers' reference row
    "newscale_fixed": ("fixed", r_newscale),   # candidate A: one-line fix, same configuration
    "rw_fixed": ("fixed", r_rw),               # candidate B: fix + no family widening at <=5 BD
    "rw_t8_fixed": ("fixed", r_rw_t8),
    "rw_t5_fixed": ("fixed", r_rw_t5),
    "ns_x0.7_fixed": ("fixed", r_ns_x07),
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
    row = {"unit": uid, "family": fam, "ptype": ptype, "cells": len(cells),
           "horizons": sorted({h for _, h in cells}), "has_h2": any(h == 2 for _, h in cells),
           "div": list(div), "scores": {}, "raw": {}, "components_n": {}}
    for name, (eng, rule) in RULES.items():
        S, c2, info = ENG[eng].draws(card, pdir, variant=rule(fam) or None)
        assert c2 == cells
        comp = R.composite(S, y, card)
        sc = R.scaled(comp, div)
        row["scores"][name] = min(sc, R.CLIP)
        row["raw"][name] = comp["composite"]
        row["components_n"][name] = [comp["marginal"] / div[0], comp["joint"] / div[1], comp["tail"] / div[2]]
        if name == "newscale_asis":
            row["steps_asis"] = info["steps"]
    # width scan on the fixed engine (drift zero, normal)
    row["widen_scan"] = {}
    for w in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
        S, _, _ = ENG["fixed"].draws(card, pdir, variant=r_widen(w)(fam))
        comp = R.composite(S, y, card)
        row["widen_scan"][str(w)] = min(R.scaled(comp, div), R.CLIP)
    # calibration: z of the realised under the organizers' M0 normal
    sd = np.sqrt(np.diag(cov))
    row["z_m0"] = [float(v) for v in (y - mean) / sd]
    cards.append(row)

names = list(RULES)
print(f"{'unit':44s} fam ptype  n hor   " + " ".join(f"{n:>14s}" for n in names))
for r in cards:
    print(f"{r['unit']:44s} {r['family']} {r['ptype']:6s} {r['cells']:2d} {str(r['horizons']):6s}" + " ".join(f"{r['scores'][n]:14.4f}" for n in names))
mat = np.array([[r["scores"][n] for n in names] for r in cards])
print(f"{'MEAN (clip 8)':66s}" + " ".join(f"{v:14.4f}" for v in mat.mean(axis=0)))
print(f"{'odd half':66s}" + " ".join(f"{v:14.4f}" for v in mat[0::2].mean(axis=0)))
print(f"{'even half':66s}" + " ".join(f"{v:14.4f}" for v in mat[1::2].mean(axis=0)))
print(f"{'MEDIAN':66s}" + " ".join(f"{v:14.4f}" for v in np.median(mat, axis=0)))
groups = {}
for key in ("family", "ptype", "has_h2"):
    for g in sorted({r[key] for r in cards}, key=str):
        idx = [i for i, r in enumerate(cards) if r[key] == g]
        groups[f"{key}={g}"] = {"n": len(idx), **{n: float(v) for n, v in zip(names, mat[idx].mean(axis=0))}}
        print(f"{key + '=' + str(g) + f' (n={len(idx)})':66s}" + " ".join(f"{v:14.4f}" for v in mat[idx].mean(axis=0)))

rng = np.random.default_rng(0)
boot_idx = rng.integers(0, len(cards), size=(20000, len(cards)))
summary = {}
for ref in ("newscale_asis", "newscale_fixed"):
    j_ref = names.index(ref)
    print(f"\nvs {ref}: mean diff, cards better/worse, LOO range of mean diff, bootstrap P(diff<0)")
    summary[ref] = {}
    for j, n in enumerate(names):
        if n == ref:
            continue
        d = mat[:, j] - mat[:, j_ref]
        loo = [np.delete(d, i).mean() for i in range(len(d))]
        boot = d[boot_idx].mean(axis=1)
        summary[ref][n] = {"mean_diff": float(d.mean()), "odd_diff": float(d[0::2].mean()), "even_diff": float(d[1::2].mean()),
                           "better": int((d < -1e-9).sum()), "worse": int((d > 1e-9).sum()),
                           "loo_min": float(min(loo)), "loo_max": float(max(loo)), "p_boot_better": float((boot < 0).mean()),
                           "largest_single_card_gain": float(d.min()), "largest_single_card_loss": float(d.max())}
        s = summary[ref][n]
        print(f"  {n:14s} diff {s['mean_diff']:+.4f} (odd {s['odd_diff']:+.4f} even {s['even_diff']:+.4f})  better {s['better']:2d} worse {s['worse']:2d}  "
              f"LOO [{s['loo_min']:+.4f},{s['loo_max']:+.4f}]  P(better) {s['p_boot_better']:.2f}  max gain {s['largest_single_card_gain']:+.3f} loss {s['largest_single_card_loss']:+.3f}")

print("\nWidth scan (fixed engine, drift zero, normal innovations), mean clipped score:")
ws = sorted(cards[0]["widen_scan"], key=float)
print(f"{'group':20s}" + " ".join(f"{w:>7s}" for w in ws))
for key in ("all", "family=F1", "family=F2", "family=F3", "family=F4", "ptype=rates", "ptype=em_fx", "ptype=g10_fx"):
    if key == "all":
        idx = list(range(len(cards)))
    else:
        k, g = key.split("=")
        idx = [i for i, r in enumerate(cards) if r[k] == g]
    vals = [np.mean([cards[i]["widen_scan"][w] for i in idx]) for w in ws]
    print(f"{key + f' (n={len(idx)})':20s}" + " ".join(f"{v:7.3f}" for v in vals))

z = np.array([v for r in cards for v in r["z_m0"]])
labels = [(r["family"], r["ptype"]) for r in cards for _ in r["z_m0"]]
print(f"\nCalibration of the organizers' M0 width at 1-4 BD (z = realised - M0 mean, in M0 sd): n={len(z)}")
print(f"  all        mean|z| {np.abs(z).mean():.3f} (normal 0.798)  rms {np.sqrt((z**2).mean()):.3f} (1.0)  in50 {(np.abs(z) < .6745).mean():.2f}  in90 {(np.abs(z) < 1.645).mean():.2f}")
calib = {"all": {"n": int(len(z)), "mean_abs_z": float(np.abs(z).mean()), "rms_z": float(np.sqrt((z**2).mean())),
                 "in50": float((np.abs(z) < .6745).mean()), "in90": float((np.abs(z) < 1.645).mean())}}
for k in (0, 1):
    for g in sorted({l[k] for l in labels}):
        zz = np.array([v for v, l in zip(z, labels) if l[k] == g])
        calib[g] = {"n": int(len(zz)), "mean_abs_z": float(np.abs(zz).mean()), "rms_z": float(np.sqrt((zz**2).mean())),
                    "in50": float((np.abs(zz) < .6745).mean()), "in90": float((np.abs(zz) < 1.645).mean())}
        print(f"  {g:10s} n={len(zz):2d} mean|z| {np.abs(zz).mean():.3f}  rms {np.sqrt((zz**2).mean()):.3f}  in50 {(np.abs(zz) < .6745).mean():.2f}  in90 {(np.abs(zz) < 1.645).mean():.2f}")

json.dump({"cards": cards, "names": names, "means": {n: float(v) for n, v in zip(names, mat.mean(axis=0))},
           "odd": {n: float(v) for n, v in zip(names, mat[0::2].mean(axis=0))},
           "even": {n: float(v) for n, v in zip(names, mat[1::2].mean(axis=0))},
           "groups": groups, "summary": summary, "calibration": calib}, OUT.open("w"), indent=1)
print("\nwrote", OUT)
