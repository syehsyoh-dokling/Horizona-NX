"""Per-card new-scale breakdown of exact M0: components ratio (marginal, joint, tail), cells, family.
Usage: new_scale_cards.py [top_n]"""
import sys
import numpy as np
import t2_core as C
import t2_sim as T
import new_scale as N
truth = C.load_truth()
names = T.card_set(truth, "all")
rows = []
for n in names:
    base, cells = C.org_m0_draws(n)
    y = C.truth_vector(truth, n, cells)
    card = C.card_of(n)
    cv = C.components(base, y, card, "pinball")
    div = np.array(N.new_scale(n))
    w = C.weights(card, base.shape[1])
    r = cv / div
    rows.append((n, len(cells), float(w @ r), r, div))
rows.sort(key=lambda t: -t[2])
top = int(sys.argv[1]) if len(sys.argv) > 1 else 15
for n, d, s, r, div in rows[:top]:
    print(f"{n:40s} cells {d:2d} score {s:7.3f}  ratios m {r[0]:6.2f} j {r[1]:6.2f} t {r[2]:6.2f}  div {div[0]:.4g} {div[1]:.4g} {div[2]:.4g}")
print("clipped (>8):", sum(1 for t in rows if t[2] > 8), "median", np.median([t[2] for t in rows]), "mean(clipped)", np.mean([min(t[2], 8) for t in rows]))
