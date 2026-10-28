"""Given a real recall run's forecasts (centres), find the best hedge for the Final: centre =
M0 mean + a x (run centre - M0 mean) per family, width multiplier w per family (shocks scaled
about the new centre), scored on the calibrated simulator with split-half CV.
Usage: blend_grid.py RUN_DIR"""
import itertools, pathlib, sys
import numpy as np, pandas as pd
import t2_core as C, t2_sim as T, new_scale as N, calib
from calibrate_live import RENAME
truth = C.load_truth(); names = T.card_set(truth, "all"); fam = {n: C.family(n) for n in names}
root = pathlib.Path(sys.argv[1]); inv = {v: k for k, v in RENAME.items()}
cache = {}
for n in names:
    f = root / n / "forecast.parquet"
    if not f.exists() and inv.get(n): f = root / inv[n] / "forecast.parquet"
    base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells); card = C.card_of(n)
    S0, _ = C.m0x_draws(n, None); mean = S0.mean(axis=0)
    df = pd.read_parquet(f); wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    S = np.column_stack([wide[(a, h)].to_numpy() for a, h in cells]); c = S.mean(axis=0)
    cache[n] = (S0, mean, c, y, card)
A = [0.0, 0.5, 0.75, 1.0]; W = [0.75, 1.0, 1.5, 2.0, 3.0]
grid = list(itertools.product(A, W)); table = {}
for gi, (a, w) in enumerate(grid):
    per = {}
    for n in names:
        S0, mean, c, y, card = cache[n]
        centre = mean + a * (c - mean)
        per[n] = min(calib.score_live(n, centre + (S0 - mean) * w, y, card), 8.0)
    table[gi] = per
odd, even = names[0::2], names[1::2]
def pick(sub): return {f: min(table, key=lambda i: np.mean([table[i][n] for n in sub if fam[n] == f] or [9])) for f in ("F1", "F2", "F3", "F4")}
def sc(p, sub): return np.mean([table[p[fam[n]]][n] for n in sub])
print("run as-is (a=1,w=1):", f"{np.mean(list(table[grid.index((1.0, 1.0))].values())):.4f}")
print("PER-FAMILY in-sample", f"{sc(pick(names), names):.4f}", {f: grid[i] for f, i in pick(names).items()})
print("CV odd->even", f"{sc(pick(odd), even):.4f}", "even->odd", f"{sc(pick(even), odd):.4f}")
for f in ("F1", "F2", "F3", "F4"):
    fn = [n for n in names if fam[n] == f]
    best = sorted(table, key=lambda i: np.mean([table[i][n] for n in fn]))[:3]
    print(f"  {f}:", ", ".join(f"{grid[i]} {np.mean([table[i][n] for n in fn]):.3f}" for i in best))
