"""Agreement gate on the NEW scale: per cell, spread = std of the K recalled points / M0 sd.
spread < g1: centre -> trimmed median, width x w1; spread < g2: centre -> median, width x w2;
else: centre stays M0, width x w_far (also when no reading). No +-1 sd clip on the move.
Grid over (g1, w1, g2, w2, w_far); split-half CV. Usage: recall_gate.py recallk.json recall3.json"""
import itertools, json, pathlib, sys
import numpy as np
import t2_core as C, t2_sim as T, new_scale as N
truth = C.load_truth(); names = T.card_set(truth, "all"); fam = {n: C.family(n) for n in names}
raws = [json.loads(pathlib.Path(p).read_text()) for p in sys.argv[1:] if pathlib.Path(p).exists()]


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
    info = {}
    for j, (a, h) in enumerate(cells):
        v = []
        for r in readings(n):
            e = r["cells"].get(f"{a}@{h}") or {}
            if str(e.get("memory", "")).lower() == "none":
                continue
            try: p = float(e["point"])
            except (KeyError, TypeError, ValueError): continue
            if np.isfinite(p) and abs(p - mean[j]) <= 8 * sd[j]: v.append(p)
        if len(v) >= 2:
            v = sorted(v); k = len(v)
            core = v[1:-1] if k >= 5 else v
            info[j] = (float(np.median(core)), float(np.std(v) / max(sd[j], 1e-12)), k)
        elif v:
            info[j] = (v[0], 9.9, 1)
    cache[n] = (S, mean, sd, cells, info, C.truth_vector(truth, n, cells), C.card_of(n))


def run(g1, w1, g2, w2, w_far):
    per = {}
    for n in names:
        S, mean, sd, cells, info, y, card = cache[n]
        centre = mean.copy(); wm = np.full(len(mean), w_far)
        for j, (p, spread, k) in info.items():
            if spread < g1: centre[j] = p; wm[j] = w1
            elif spread < g2: centre[j] = p; wm[j] = w2
        per[n] = min(N.score_new(n, centre + (S - mean) * wm, y, card), N.CLIP)
    return per


best = None
for g1, w1, g2, w2, w_far in itertools.product([0.25, 0.5, 0.75], [0.5, 0.75, 1.0], [1.0, 1.5, 2.5, 99], [1.0, 1.5], [1.5, 2.0, 3.0]):
    per = run(g1, w1, g2, w2, w_far); m = np.mean(list(per.values()))
    if best is None or m < best[0]: best = (m, (g1, w1, g2, w2, w_far), per)
m, cfg, per = best
odd = np.mean([per[n] for n in names[0::2]]); even = np.mean([per[n] for n in names[1::2]])
print("BEST gate (g1,w1,g2,w2,w_far) =", cfg, f"mean {m:.4f} halves {odd:.3f}/{even:.3f}",
      " ".join(f"{f} {np.mean([per[n] for n in names if fam[n]==f]):.3f}" for f in ("F1","F2","F3","F4")), "clipped", sum(1 for v in per.values() if v >= 8))
for cfg2 in [(0.5, 0.6, 1.5, 1.0, 2.0), (0.5, 0.75, 1.5, 1.0, 2.0), (0.5, 1.0, 99, 1.0, 3.0)]:
    per = run(*cfg2); print("  ", cfg2, f"{np.mean(list(per.values())):.4f}")
print("cells with >=2 readings:", sum(1 for n in names for j, v in cache[n][4].items() if v[2] >= 2), "of", sum(len(cache[n][3]) for n in names))
