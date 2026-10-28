"""Per-family choice of a statistical variant, with cross-validation (text off, 71 scored cards).

For each family the candidate with the lowest mean on one half of its cards is scored on the OTHER
half (both directions); that cross-validated number is what a per-family configuration can be
expected to deliver on cards it was not chosen on. The in-sample best is printed next to it.
"""
import json

import numpy as np

import t2_core as C
import t2_sim as T

CANDS = {
    "t8": {"t_nu": 8.0},
    "d075": {"t_nu": 8.0, "drift": 0.75},
    "d05": {"t_nu": 8.0, "drift": 0.5},
    "d025": {"t_nu": 8.0, "drift": 0.25},
    "d0": {"t_nu": 8.0, "drift": "zero"},
    "dt1": {"t_nu": 8.0, "drift": "t1"},
    "d05_ewma": {"t_nu": 8.0, "drift": 0.5, "ewma": [0.97, 0.5]},
    "d075_ewma": {"t_nu": 8.0, "drift": 0.75, "ewma": [0.97, 0.5]},
    "ewma": {"t_nu": 8.0, "ewma": [0.97, 0.5]},
    "d05_t5": {"t_nu": 5.0, "drift": 0.5},
    "d05_w095": {"t_nu": 8.0, "drift": 0.5, "widen": 0.95},
    "d05_w11": {"t_nu": 8.0, "drift": 0.5, "widen": 1.1},
}
truth = C.load_truth()
names = T.card_set(truth, "all")
scores = {k: T.simulate(v, "m0x", names, truth) for k, v in CANDS.items()}
clip = lambda x: min(x, C.CLIP)  # noqa: E731
json.dump(scores, open(T.HERE / "family_cv_scores.json", "w"), indent=1)

print("variant      " + "  ".join(f"{f:>6s}" for f in T.FAMS) + "     all")
for k in CANDS:
    fam = [np.mean([clip(scores[k][n]) for n in names if C.family(n) == f]) for f in T.FAMS]
    print(f"{k:12s} " + "  ".join(f"{v:6.4f}" for v in fam) + f"  {np.mean([clip(scores[k][n]) for n in names]):6.4f}")

total_in, total_cv, n_all = 0.0, 0.0, 0
print("\nfamily  n   in-sample best        cross-validated (pick on one half, score the other)")
for f in T.FAMS:
    cards = sorted(n for n in names if C.family(n) == f)
    a, b = cards[0::2], cards[1::2]
    mean = lambda k, cs: float(np.mean([clip(scores[k][n]) for n in cs]))  # noqa: E731
    best = min(CANDS, key=lambda k: mean(k, cards))
    pa, pb = min(CANDS, key=lambda k: mean(k, a)), min(CANDS, key=lambda k: mean(k, b))
    cv = (mean(pa, b) * len(b) + mean(pb, a) * len(a)) / len(cards)
    print(f"{f}     {len(cards):2d}  {best:10s} {mean(best, cards):.4f}    {cv:.4f}  (picks {pa} / {pb})")
    total_in += mean(best, cards) * len(cards); total_cv += cv * len(cards); n_all += len(cards)
print(f"\nper-family configuration: in-sample {total_in / n_all:.4f}, cross-validated {total_cv / n_all:.4f}")
print(f"one global setting (d05): {np.mean([clip(scores['d05'][n]) for n in names]):.4f}")
