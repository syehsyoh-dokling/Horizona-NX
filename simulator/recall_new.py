"""Recall on the NEW scale: centre moved a fraction a toward the recalled median point (K readings,
recallk.json), width = w x M0's sd (Student-t nu optional), scored with the expected-error divisor,
clip 8. Grid per family + split-half CV. Usage: recall_new.py [recallk.json ...]"""
import itertools, json, os, pathlib, sys
import numpy as np
import t2_core as C, t2_sim as T, new_scale as N
truth = C.load_truth(); names = T.card_set(truth, "all")
fam = {n: C.family(n) for n in names}
paths = [pathlib.Path(p) for p in (sys.argv[1:] or ["recallk.json"])]
raws = [json.loads(p.read_text()) for p in paths if p.exists()]


def readings(n):
    out = []
    for raw in raws:
        r = raw.get(n); r = [r] if isinstance(r, dict) else (r or [])
        out.extend(x for x in r if isinstance(x, dict) and "cells" in x)
    return out


cache = {}
for n in names:
    S, cells = C.m0x_draws(n, None)            # exact M0 draws (mean + shocks)
    mean, cov, _ = N.m0_mean_cov(n)
    sd = np.sqrt(np.diag(cov))
    pts = {}
    for j, (a, h) in enumerate(cells):
        v = []
        for r in readings(n):
            e = r["cells"].get(f"{a}@{h}") or {}
            if str(e.get("memory", "")).lower() == "none":
                continue
            try:
                p = float(e["point"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(p) and abs(p - mean[j]) <= 8 * sd[j]:
                v.append(p)
        if v:
            pts[j] = float(np.median(v))
    cache[n] = (S, mean, cells, pts, C.truth_vector(truth, n, cells), C.card_of(n))

A = [0.0, 0.5, 0.75, 1.0]; W = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
grid = list(itertools.product(A, W)); table = {}
for gi, (a, w) in enumerate(grid):
    per = {}
    for n in names:
        S, mean, cells, pts, y, card = cache[n]
        shocks = (S - mean) * w
        centre = mean.copy()
        for j, p in pts.items():
            centre[j] = (1 - a) * mean[j] + a * p
        per[n] = min(N.score_new(n, centre + shocks, y, card), N.CLIP)
    table[gi] = per
    print(f"a={a:.2f} w={w:.2f} mean {np.mean(list(per.values())):.4f} " + " ".join(f"{f} {np.mean([per[n] for n in names if fam[n]==f]):.3f}" for f in ("F1","F2","F3","F4")), flush=True)
odd, even = names[0::2], names[1::2]
def pick(sub):
    return {f: min(table, key=lambda i: np.mean([table[i][n] for n in sub if fam[n] == f] or [9])) for f in ("F1","F2","F3","F4")}
def sc(picks, sub): return np.mean([table[picks[fam[n]]][n] for n in sub])
print("PER-FAMILY in-sample", f"{sc(pick(names), names):.4f}", {f: grid[i] for f, i in pick(names).items()})
print("CV odd->even", f"{sc(pick(odd), even):.4f}", " even->odd", f"{sc(pick(even), odd):.4f}")
print("cards with a recalled point:", sum(1 for n in names if cache[n][3]), "/", len(names))
