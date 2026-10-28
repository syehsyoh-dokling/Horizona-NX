"""Unexplored lever on the NEW scale: distance-adaptive width. Centre = recalled point; width
multiplier per cell = clip(|point - M0 mean| / sd / c, wmin, wmax): a recall that disagrees with
M0 by many sd is where M0's tail would be clipped at 8, so widen there. Also the oracle ceiling
(centre = truth) for reference. Usage: recall_adapt.py [recallk.json ...]"""
import itertools, json, pathlib, sys
import numpy as np
import t2_core as C, t2_sim as T, new_scale as N
truth = C.load_truth(); names = T.card_set(truth, "all"); fam = {n: C.family(n) for n in names}
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
    S, cells = C.m0x_draws(n, None)
    mean, cov, _ = N.m0_mean_cov(n); sd = np.sqrt(np.diag(cov))
    pts = {}
    for j, (a, h) in enumerate(cells):
        v = []
        for r in readings(n):
            e = r["cells"].get(f"{a}@{h}") or {}
            if str(e.get("memory", "")).lower() == "none":
                continue
            try: p = float(e["point"])
            except (KeyError, TypeError, ValueError): continue
            if np.isfinite(p) and abs(p - mean[j]) <= 8 * sd[j]: v.append(p)
        if v: pts[j] = float(np.median(v))
    cache[n] = (S, mean, sd, cells, pts, C.truth_vector(truth, n, cells), C.card_of(n))


def run(c, wmin, wmax, oracle=False):
    per = {}
    for n in names:
        S, mean, sd, cells, pts, y, card = cache[n]
        centre = mean.copy(); wm = np.full(len(mean), wmin)
        for j in range(len(mean)):
            p = y[j] if oracle else pts.get(j)
            if p is None: continue
            centre[j] = p
            wm[j] = min(max(abs(p - mean[j]) / max(sd[j], 1e-12) / c, wmin), wmax)
        per[n] = min(N.score_new(n, centre + (S - mean) * wm, y, card), N.CLIP)
    return per


print("oracle (centre = truth, M0 width):", f"{np.mean(list(run(1e9, 1, 1, True).values())):.4f}")
print("oracle (centre = truth, width x0.5):", f"{np.mean(list(run(1e9, 0.5, 0.5, True).values())):.4f}")
best = None
for c, wmin, wmax in itertools.product([1.0, 2.0, 3.0, 4.0], [0.75, 1.0, 1.5], [1.5, 2.0, 3.0, 5.0]):
    if wmax < wmin: continue
    per = run(c, wmin, wmax); m = np.mean(list(per.values()))
    odd = np.mean([per[n] for n in names[0::2]]); even = np.mean([per[n] for n in names[1::2]])
    if best is None or m < best[0]: best = (m, c, wmin, wmax, odd, even, per)
    print(f"c={c} wmin={wmin} wmax={wmax} mean {m:.4f} halves {odd:.3f}/{even:.3f} clipped {sum(1 for v in per.values() if v >= 8)}")
m, c, wmin, wmax, odd, even, per = best
print(f"BEST c={c} wmin={wmin} wmax={wmax} mean {m:.4f} halves {odd:.3f}/{even:.3f} " + " ".join(f"{f} {np.mean([per[n] for n in names if fam[n]==f]):.3f}" for f in ("F1","F2","F3","F4")))
