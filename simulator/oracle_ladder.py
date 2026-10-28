"""What would it take to reach 0.5? An ORACLE ladder on the 71 scored cards (dev analysis only).

Base forecast = the live best (t8 + drift x0.5, text off). Two questions:
  A. centre moved a fraction `a` of the way to the realised value, width x `w`  (needs the answer)
  B. direction-only signal: each cell's centre moved k sd in the TRUE direction with probability p,
     in the wrong direction otherwise (a forecaster that knows only up/down, right p of the time)
Neither is a method: A reads the truth, B assumes a hit rate. They size the skill 0.5 demands.
"""
import numpy as np

import t2_core as C
import t2_sim as T

truth = C.load_truth()
names = T.card_set(truth, "all")
BASE_VARIANT = {"t_nu": 8.0, "drift": 0.5}
cache = {}
for n in names:
    base, cells = T.base_of(n)
    S, _ = C.m0x_draws(n, BASE_VARIANT)
    cache[n] = (base, S, C.truth_vector(truth, n, cells), C.card_of(n))


def primary(transform):
    vals = [min(C.card_score(card, base, transform(n, S, y), y), C.CLIP)
            for n, (base, S, y, card) in cache.items()]
    return float(np.mean(vals))


def shift_to_truth(a, w):
    def f(_n, S, y):
        c = S.mean(axis=0)
        return (c + a * (y - c)) + (S - c) * w
    return f


def direction(p, k, seed):
    rng = np.random.default_rng(seed)

    def f(_n, S, y):
        c, sd = S.mean(axis=0), S.std(axis=0)
        right = rng.random(c.shape) < p
        step = np.sign(y - c) * np.where(right, 1.0, -1.0) * k * sd
        return S + step
    return f


print("A. centre moved a fraction a to the truth, width x w  (primary, lower is better)")
print("    a \\ w   1.00    0.75    0.50    0.25")
for a in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    print(f"   {a:4.2f}   " + "  ".join(f"{primary(shift_to_truth(a, w)):.4f}" for w in (1.0, 0.75, 0.5, 0.25)))

print("\nB. direction-only signal, centre moved k sd (mean of 5 seeds)")
print("    p \\ k   0.25    0.50    1.00")
for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    row = []
    for k in (0.25, 0.5, 1.0):
        row.append(np.mean([primary(direction(p, k, s)) for s in range(5)]))
    print(f"   {p:4.2f}   " + "  ".join(f"{v:.4f}" for v in row))
