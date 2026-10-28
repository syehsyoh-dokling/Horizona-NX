"""Grid over text-off M0 variants (widen, t_nu, drift) on the NEW scale (expected-error divisor,
clip 8). Global best, per-family best, and a split-half cross-validation (pick on odd cards,
score on even and vice versa). Usage: grid_new.py"""
import itertools
import numpy as np
import t2_core as C
import t2_sim as T
import new_scale as N

truth = C.load_truth()
names = T.card_set(truth, "all")
fam = {n: C.family(n) for n in names}
WIDEN = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
TNU = [None, 8.0, 5.0, 3.0]
DRIFT = ["m0", 0.5, "zero"]
cache = {}


def score_card(n, variant):
    key = (n, tuple(sorted((k, str(v)) for k, v in variant.items())))
    if key not in cache:
        s, cells = C.m0x_draws(n, variant or None)
        y = C.truth_vector(truth, n, cells)
        cache[key] = min(N.score_new(n, s, y), N.CLIP)
    return cache[key]


def variant_of(w, nu, dr):
    v = {}
    if w != 1.0:
        v["widen"] = w
    if nu:
        v["t_nu"] = nu
    if dr != "m0":
        v["drift"] = dr
    return v


grid = [variant_of(w, nu, dr) for w, nu, dr in itertools.product(WIDEN, TNU, DRIFT)]
table = {}   # variant index -> {name: score}
for i, v in enumerate(grid):
    table[i] = {n: score_card(n, v) for n in names}
    print(f"{i:3d} {str(v):45s} mean {np.mean(list(table[i].values())):.4f}", flush=True)

means = {i: np.mean(list(t.values())) for i, t in table.items()}
best = min(means, key=means.get)
print("\nGLOBAL BEST", grid[best], f"{means[best]:.4f}")
for f in ("F1", "F2", "F3", "F4"):
    fn = [n for n in names if fam[n] == f]
    fb = min(table, key=lambda i: np.mean([table[i][n] for n in fn]))
    print(f"  {f} best {grid[fb]} {np.mean([table[fb][n] for n in fn]):.4f}  (global-best variant gives {np.mean([table[best][n] for n in fn]):.4f})")
# per-family mix, cross-validated by halves
odd = names[0::2]; even = names[1::2]
def pick(sub):
    out = {}
    for f in ("F1", "F2", "F3", "F4"):
        fn = [n for n in sub if fam[n] == f]
        out[f] = min(table, key=lambda i: np.mean([table[i][n] for n in fn])) if fn else best
    return out
def score_with(picks, sub):
    return np.mean([table[picks[fam[n]]][n] for n in sub])
p_odd, p_even = pick(odd), pick(even)
print("\nPER-FAMILY MIX in-sample", f"{score_with(pick(names), names):.4f}",
      "| CV: pick odd->score even", f"{score_with(p_odd, even):.4f}", " pick even->score odd", f"{score_with(p_even, odd):.4f}")
print("  picks (all cards):", {f: grid[i] for f, i in pick(names).items()})
print("  picks odd :", {f: grid[i] for f, i in p_odd.items()})
print("  picks even:", {f: grid[i] for f, i in p_even.items()})
