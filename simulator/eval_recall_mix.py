"""How close can model memory get, locally? (7 Oct, owner's question: a configuration near 0.6)

recall.json / recallk.json (K readings per card, memory allowed) -> per cell the median recalled
point. Grid: per family the weight a_w of the recalled point in the centre (0 = base, 1 = recall),
with the base width, a widened width, or the model's band; then the best per-family mix, its two
split halves, and the oracle ladder (centre a fraction f of the way to the TRUTH) for reference.
Usage: eval_recall_mix.py [recallk.json]
"""
import json
import os
import pathlib
import sys

import numpy as np

import t2_core as C
import t2_sim as T

truth = C.load_truth()
names = T.card_set(truth, "all")
FAMS = ("F1", "F2", "F3", "F4")
BASE = {"t_nu": 8.0, "drift": 0.5}
cache = {}
for n in names:
    base, cells = T.base_of(n)
    S, _ = C.m0x_draws(n, BASE)
    cache[n] = (base, S, C.truth_vector(truth, n, cells), C.card_of(n), cells)

paths = [pathlib.Path(p) for p in (sys.argv[1:] or ["recallk.json"])]
raws = [json.loads(p.read_text()) for p in paths if p.exists()]
print("reading files:", ", ".join(str(p) for p in paths))


def readings(n):
    out = []
    for raw in raws:
        r = raw.get(n)
        if isinstance(r, dict):
            r = [r]
        out.extend(x for x in (r or []) if isinstance(x, dict) and "cells" in x)
    return out


def recalled(n, cells, S):
    """{j: (median point, median half-width)} over the K readings, unit slips dropped."""
    c, sd = S.mean(axis=0), S.std(axis=0)
    out = {}
    mode = os.environ.get("RECALL_MODE", "point")      # point | sd (direction x size_sd, recall3)
    for j, (a, h) in enumerate(cells):
        pts, halves = [], []
        for r in readings(n):
            e = r["cells"].get(f"{a}@{h}") or {}
            if str(e.get("memory", "")).lower() == "none":
                continue
            try:
                if mode == "sd" and "size_sd" in e:
                    pt = float(c[j] + float(e.get("direction", 0)) * min(6.0, abs(float(e["size_sd"]))) * sd[j])
                else:
                    pt = float(e["point"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(pt) or abs(pt - c[j]) > 8 * sd[j]:
                continue
            pts.append(pt)
            try:
                half = (float(e["hi"]) - float(e["lo"])) / 2
                if half > 0:
                    halves.append(half)
            except (KeyError, TypeError, ValueError):
                pass
        if pts:
            out[j] = (float(np.median(pts)), float(np.median(halves)) if halves else None)
    return out


REC = {n: recalled(n, cache[n][4], cache[n][1]) for n in names}
recog = {n for n in names if any(r.get("recognised") is True for r in readings(n))}
print(f"cards with a recalled point: {sum(1 for n in names if REC[n])}/{len(names)}; recognised {len(recog)}")


def score(n, S2):
    base, S, y, card, cells = cache[n]
    return min(C.card_score(card, base, S2, y), C.CLIP)


def shifted(n, a_w, width):
    base, S, y, card, cells = cache[n]
    c, sd = S.mean(axis=0), S.std(axis=0)
    out = S.copy()
    for j, (pt, half) in REC[n].items():
        m = 1.0
        if width == "band" and half:
            m = max(0.4, min(1.6, (half / 1.645) / sd[j]))
        elif isinstance(width, float):
            m = width
        out[:, j] = c[j] + a_w * (pt - c[j]) + (S[:, j] - c[j]) * m
    return out


def oracle(n, f):
    base, S, y, card, cells = cache[n]
    c = S.mean(axis=0)
    return S + f * (y - c)


grid_w = [0.0, 0.25, 0.5, 0.75, 1.0]
grid_width = ["base", 1.25, 1.5, "band"]
table = {}   # (a_w, width) -> {card: score}
for a_w in grid_w:
    for wd in grid_width:
        table[(a_w, wd)] = {n: score(n, shifted(n, a_w, wd)) for n in names}

print("\nGLOBAL (all families same setting): mean score")
print("  a_w   " + "  ".join(f"{str(w):>6}" for w in grid_width))
for a_w in grid_w:
    print(f"  {a_w:4.2f}  " + "  ".join(f"{np.mean(list(table[(a_w, wd)].values())):6.4f}" for wd in grid_width))

print("\nBY FAMILY: best setting (a_w, width) and its family mean")
best = {}
for f in FAMS:
    fam_names = [n for n in names if C.family(n) == f]
    cand = {k: float(np.mean([v[n] for n in fam_names])) for k, v in table.items()}
    k = min(cand, key=cand.get)
    best[f] = k
    print(f"  {f}: base {cand[(0.0, 'base')]:.3f} -> best {cand[k]:.3f} at a_w={k[0]}, width={k[1]}")

mixed = {n: table[best[C.family(n)]][n] for n in names}
odd = [mixed[n] for i, n in enumerate(sorted(names)) if i % 2 == 0]
even = [mixed[n] for i, n in enumerate(sorted(names)) if i % 2 == 1]
print(f"\nBEST PER-FAMILY MIX: mean {np.mean(list(mixed.values())):.4f}   odd half {np.mean(odd):.4f}   even half {np.mean(even):.4f}")
print("  (the mix is chosen on the same cards it is scored on: an optimistic number)")

# honest cross-check: choose the family setting on the odd half, score on the even half, and vice versa
def cv_mix():
    srt = sorted(names)
    halves = [srt[0::2], srt[1::2]]
    out = {}
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        for f in FAMS:
            fit_f = [n for n in fit if C.family(n) == f]
            if not fit_f:
                continue
            k = min(table, key=lambda k: np.mean([table[k][n] for n in fit_f]))
            for n in test:
                if C.family(n) == f:
                    out[n] = table[k][n]
    return out


cv = cv_mix()
print(f"CROSS-VALIDATED MIX (setting chosen on the other half): mean {np.mean(list(cv.values())):.4f}")

print("\nORACLE LADDER (centre moved a fraction f towards the truth, base width):")
for f in (0.25, 0.5, 0.75, 1.0):
    print(f"  f={f:.2f}: {np.mean([score(n, oracle(n, f)) for n in names]):.4f}")

# where the memory is right / wrong
print("\nRECALLED POINT QUALITY by family: cells, direction right, median |err| base -> recall (in sd)")
for f in FAMS:
    eb, er, right, tot = [], [], 0, 0
    for n in names:
        if C.family(n) != f:
            continue
        base, S, y, card, cells = cache[n]
        c, sd = S.mean(axis=0), S.std(axis=0)
        for j, (pt, _half) in REC[n].items():
            eb.append(abs(y[j] - c[j]) / sd[j]); er.append(abs(y[j] - pt) / sd[j])
            tot += 1; right += int(np.sign(pt - c[j]) == np.sign(y[j] - c[j]))
    if tot:
        print(f"  {f}: {tot:3d} cells, right {right / tot:.0%}, |err| {np.median(eb):.2f} -> {np.median(er):.2f}")
