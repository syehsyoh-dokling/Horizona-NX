"""Pre-registered variants A-D on top of newscalegate (gate5), scored on the 22 -ps1 cards (26 Oct).
Decision rule fixed BEFORE the run: a variant replaces gate5 only if it is better on BOTH halves (odd / even
cards), every leave-one-out mean difference is < 0, and the bootstrap P(better) >= 0.8.
Variants (theory, not data search):
  v1_ewma94     RiskMetrics daily EWMA (lambda 0.94) volatility, correlations kept
  v2_ewma94_50  50 % EWMA(0.94) variance + 50 % M0's 300-day variance (vol mean reversion, GARCH-like)
  v3_ewma97_50  50 % EWMA(0.97) + 50 % M0 (slower, more conservative)
usage: compare_ewma_ps1.py <inputs_dir> <answers_dir> <out_json>
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


GATE = load_engine(HERE / "agent_vr" / "forecast_core" / "m0_exact.py", "m0_vr")
NEWSCALE = {
    "F1": {"t_nu": 5.0, "drift": "zero"},
    "F2": {"widen": 1.5, "drift": "zero"},
    "F3": {"widen": 1.25, "t_nu": 5.0, "drift": "zero"},
    "F4": {"widen": 3.0},
}


def gate(fam, **extra):
    v = dict(NEWSCALE[fam])
    v["widen_gate"] = 5.0
    v.update(extra)
    return v


RULES = {
    "gate5": lambda f: gate(f),
    "A_t5_all": lambda f: gate(f, t_nu=5.0),
    "B_normal_all": lambda f: {k: v for k, v in gate(f).items() if k != "t_nu"},
    "C_zero_drift": lambda f: gate(f, drift="zero"),
    "D_vr": lambda f: gate(f, vr=(0.5, 2.0)),
    "m0": lambda f: None,
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
    row = {"unit": uid, "family": fam, "scores": {}}
    for name, rule in RULES.items():
        S, c2, _ = GATE.draws(card, pdir, variant=rule(fam) or None)
        assert c2 == cells
        row["scores"][name] = min(R.scaled(R.composite(S, y, card), div), R.CLIP)
    cards.append(row)

names = list(RULES)
mat = np.array([[r["scores"][n] for n in names] for r in cards])
print(f"{'unit':44s}" + " ".join(f"{n:>13s}" for n in names))
for r in cards:
    print(f"{r['unit']:44s}" + " ".join(f"{r['scores'][n]:13.4f}" for n in names))
for lab, sl in (("MEAN", slice(None)), ("odd half", slice(0, None, 2)), ("even half", slice(1, None, 2))):
    print(f"{lab:44s}" + " ".join(f"{v:13.4f}" for v in mat[sl].mean(axis=0)))

rng = np.random.default_rng(0)
bidx = rng.integers(0, len(cards), size=(20000, len(cards)))
j0 = names.index("gate5")
summary = {}
print("\nvs gate5 (decision rule: odd<0, even<0, LOO max<0, P>=0.8)")
for j, n in enumerate(names):
    if n == "gate5":
        continue
    d = mat[:, j] - mat[:, j0]
    loo = [np.delete(d, i).mean() for i in range(len(d))]
    s = {"mean": float(d.mean()), "odd": float(d[0::2].mean()), "even": float(d[1::2].mean()),
         "loo_max": float(max(loo)), "loo_min": float(min(loo)), "p": float((d[bidx].mean(axis=1) < 0).mean()),
         "better": int((d < -1e-9).sum()), "worse": int((d > 1e-9).sum())}
    s["PASS"] = bool(s["odd"] < 0 and s["even"] < 0 and s["loo_max"] < 0 and s["p"] >= 0.8)
    summary[n] = s
    print(f"  {n:13s} diff {s['mean']:+.4f} odd {s['odd']:+.4f} even {s['even']:+.4f} LOO [{s['loo_min']:+.4f},"
          f"{s['loo_max']:+.4f}] P {s['p']:.2f} better/worse {s['better']}/{s['worse']}  -> {'PASS' if s['PASS'] else 'fail'}")
json.dump({"cards": cards, "names": names, "summary": summary}, OUT.open("w"), indent=1)
print("wrote", OUT)
