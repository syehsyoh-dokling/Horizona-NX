"""Which M0 definition produces the organizer's divisors? For candidate trailing windows W (and
full history), recompute the expected-error divisors and the newscale score per card; compare
with the live per-card scores (ratio spread). The right definition gives ratio ~1 with a tight
spread. Usage: window_probe.py"""
import numpy as np
import t2_core as C, t2_sim as T, new_scale as N
from forecast_core import m0_exact as M
from calibrate_live import RENAME, NEWSCALE, live
truth = C.load_truth(); names = T.card_set(truth, "all")
print("current WINDOW =", getattr(M, "WINDOW", None))
cv = {}
for n in names:
    base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells)
    s, _ = C.m0x_draws(n, NEWSCALE[C.family(n)])
    cv[n] = (C.components(s, y, C.card_of(n), "pinball"), C.weights(C.card_of(n), len(cells)), live[RENAME.get(n, n)])
orig = M.WINDOW
for W in (orig, 250, 500, 504, 756, 1000, 1260, 100000):
    M.WINDOW = W; N._CACHE.clear()
    r = []
    for n in names:
        comp, w, lv = cv[n]
        div = np.array(N.new_scale(n))
        r.append(lv / float(w @ (comp / div)))
    r = np.array(r)
    print(f"W={W:>6}: live/sim median {np.median(r):.3f}  IQR {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f}  log-sd {np.std(np.log(r)):.3f}")
M.WINDOW = orig; N._CACHE.clear()
