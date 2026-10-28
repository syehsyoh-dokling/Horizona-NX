"""Stress-gated width on top of newscale: a pre-outcome, data-only feature per card,
r = std(last 21 steps) / std(all history steps) (recent vs long-run volatility, averaged over the
card's assets), decides an extra width multiplier e when r >= thr (and optionally a narrowing when
r <= 1/thr). Scored on the calibrated simulator; split-half CV; shuffled-r downside test.
Usage: stress_gate.py"""
import itertools
import numpy as np, pandas as pd
import t2_core as C, t2_sim as T, calib
from calibrate_live import NEWSCALE
from forecast_core import m0_exact as M
truth = C.load_truth(); names = T.card_set(truth, "all"); fam = {n: C.family(n) for n in names}
cache = {}; feat = {}
for n in names:
    card = C.card_of(n); asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    ttype = str(card["targets"].get("target_type", "level"))
    rs = []
    for a in card["targets"]["asset_ids"]:
        try:
            h = M.history(C.UNITS / n, str(a), asof); st = np.asarray(M.steps(h, ttype), float); st = st[np.isfinite(st)]
            if len(st) > 60: rs.append(np.std(st[-21:]) / (np.std(st) + 1e-12))
        except Exception:
            pass
    feat[n] = float(np.mean(rs)) if rs else 1.0
    S0, cells = C.m0x_draws(n, NEWSCALE[fam[n]]); mean = S0.mean(axis=0)
    y = C.truth_vector(truth, n, cells); cache[n] = (S0, mean, y, card)
r = np.array([feat[n] for n in names])
base = {n: min(calib.score_live(n, cache[n][0], cache[n][2], cache[n][3]), 8.0) for n in names}
b = np.array([base[n] for n in names])
print(f"newscale baseline (calibrated sim): {b.mean():.4f}; feature r: median {np.median(r):.2f} IQR {np.percentile(r,25):.2f}-{np.percentile(r,75):.2f} max {r.max():.2f}")
print(f"corr(r, score) = {np.corrcoef(r, b)[0,1]:+.3f}; mean score r>=1.25: {b[r>=1.25].mean():.2f} (n={np.sum(r>=1.25)}), r<0.8: {b[r<0.8].mean():.2f} (n={np.sum(r<0.8)})")
def score_with(feature, thr, e, lo):
    per = {}
    for i, n in enumerate(names):
        S0, mean, y, card = cache[n]; f = feature[i]
        w = e if f >= thr else (lo if f <= 1.0 / thr else 1.0)
        per[n] = min(calib.score_live(n, mean + (S0 - mean) * w, y, card), 8.0)
    return per
grid = [(thr, e, lo) for thr, e, lo in itertools.product((1.15, 1.3, 1.5, 2.0), (1.25, 1.5, 2.0, 3.0), (1.0, 0.85, 0.7))]
table = {g: score_with(r, *g) for g in grid}
odd, even = names[0::2], names[1::2]
def best(sub): return min(grid, key=lambda g: np.mean([table[g][n] for n in sub]))
g_all = best(names); print(f"best in-sample {g_all}: {np.mean(list(table[g_all].values())):.4f}  (baseline {b.mean():.4f})")
go, ge = best(odd), best(even)
print(f"CV odd->even {np.mean([table[go][n] for n in even]):.4f} (baseline even {np.mean([base[n] for n in even]):.4f}) | even->odd {np.mean([table[ge][n] for n in odd]):.4f} (baseline odd {np.mean([base[n] for n in odd]):.4f})")
rng = np.random.default_rng(0)
sh = [np.mean(list(score_with(rng.permutation(r), *g_all).values())) for _ in range(5)]
print(f"shuffled-r downside with {g_all}: mean {np.mean(sh):.4f} (min {min(sh):.3f} max {max(sh):.3f})")
print("top 5 grid points:", ", ".join(f"{g}:{np.mean(list(table[g].values())):.3f}" for g in sorted(grid, key=lambda g: np.mean(list(table[g].values())))[:5]))
