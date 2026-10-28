"""Two-component shock mixture on the calibrated simulator: a fraction p of the 500 draws gets its
shock scaled by w_wide, the rest by w_core (about M0's mean, zero drift optional). Per family,
split-half CV. Usage: mixture_grid.py"""
import itertools
import numpy as np
import t2_core as C, t2_sim as T, calib
truth = C.load_truth(); names = T.card_set(truth, "all"); fam = {n: C.family(n) for n in names}
cache = {}
for n in names:
    S0, cells = C.m0x_draws(n, {"drift": "zero"})
    cache[n] = (S0, S0.mean(axis=0), C.truth_vector(truth, n, cells), C.card_of(n))
P = [0.0, 0.1, 0.2, 0.3]; WC = [0.75, 1.0, 1.25, 1.5]; WW = [3.0, 5.0, 8.0]
grid = [(p, wc, ww) for p, wc, ww in itertools.product(P, WC, WW) if not (p == 0.0 and ww != WW[0])]
table = {}
rng_mask = {n: (np.arange(500) % 10) for n in names}     # deterministic: draws 0,10,20.. first in the wide set
for gi, (p, wc, ww) in enumerate(grid):
    per = {}
    for n in names:
        S0, mean, y, card = cache[n]
        sh = S0 - mean
        k = int(round(p * 500))
        mult = np.full(500, wc); mult[:k] = ww
        per[n] = min(calib.score_live(n, mean + sh * mult[:, None], y, card), 8.0)
    table[gi] = per
odd, even = names[0::2], names[1::2]
def pick(sub): return {f: min(table, key=lambda i: np.mean([table[i][n] for n in sub if fam[n] == f] or [9])) for f in ("F1", "F2", "F3", "F4")}
def sc(pk, sub): return np.mean([table[pk[fam[n]]][n] for n in sub])
base_i = grid.index((0.0, 1.0, 3.0))
print("zero-drift M0 (no mixture):", f"{np.mean(list(table[base_i].values())):.4f}")
print("PER-FAMILY in-sample", f"{sc(pick(names), names):.4f}", {f: grid[i] for f, i in pick(names).items()})
print("CV odd->even", f"{sc(pick(odd), even):.4f}", "even->odd", f"{sc(pick(even), odd):.4f}")
for f in ("F1", "F2", "F3", "F4"):
    fn = [n for n in names if fam[n] == f]
    best = sorted(table, key=lambda i: np.mean([table[i][n] for n in fn]))[:3]
    print(f"  {f}:", ", ".join(f"p={grid[i][0]} core={grid[i][1]} wide={grid[i][2]} -> {np.mean([table[i][n] for n in fn]):.3f}" for i in best))
pk = pick(names); print("clipped cards with the per-family pick:", sum(1 for n in names if table[pk[fam[n]]][n] >= 8.0))
