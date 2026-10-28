"""Family-specific WIDTH and tail grid (text off, no memory): the realised move on F2/F4 cards is
~3 sd of the forecast, so the distribution is far too narrow there. Per family: the best widen /
tail setting in-sample and CROSS-VALIDATED (picked on one half of the family's cards, scored on
the other), then the combined configuration.
"""
import json

import numpy as np

import t2_core as C
import t2_sim as T

BASE = {"F1": {"drift": 0.5}, "F2": {"drift": 0.5}, "F3": {"drift": 0.5}, "F4": {"drift": "t1"}}
OPTS = {}
for nu in (8.0, 4.0, 3.0):
    for w in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
        OPTS[f"t{nu:g} x{w:g}"] = {"t_nu": nu, "widen": w}
truth = C.load_truth()
names = T.card_set(truth, "all")
score = {}
for f in T.FAMS:
    cards = [n for n in names if C.family(n) == f]
    for k, o in OPTS.items():
        v = dict(BASE[f]); v.update(o)
        if v.get("widen") == 1.0:
            v.pop("widen")
        s = T.simulate(v, "m0x", cards, truth)
        for n in cards:
            score.setdefault(k, {})[n] = min(s[n], C.CLIP)
json.dump(score, open(T.HERE / "grid_width_scores.json", "w"), indent=1)

keys = list(OPTS)
print("option      " + "  ".join(f"{f:>6s}" for f in T.FAMS))
for k in keys:
    print(f"{k:10s}  " + "  ".join(f"{np.mean([score[k][n] for n in names if C.family(n) == f]):6.3f}" for f in T.FAMS))

tin = tcv = tbase = 0.0
choice = {}
print("\nfamily  n  base    in-sample best        cross-validated  (picks on the two halves)")
for f in T.FAMS:
    cards = sorted(n for n in names if C.family(n) == f)
    a, b = cards[0::2], cards[1::2]
    m = lambda k, cs: float(np.mean([score[k][n] for n in cs]))  # noqa: E731
    best = min(keys, key=lambda k: m(k, cards))
    pa, pb = min(keys, key=lambda k: m(k, a)), min(keys, key=lambda k: m(k, b))
    cv = (m(pa, b) * len(b) + m(pb, a) * len(a)) / len(cards)
    choice[f] = best
    print(f"{f}     {len(cards):2d}  {m('t8 x1', cards):.3f}   {best:9s} {m(best, cards):.3f}       {cv:.3f}   ({pa} / {pb})")
    tin += m(best, cards) * len(cards); tcv += cv * len(cards); tbase += m("t8 x1", cards) * len(cards)
N = len(names)
print(f"\nall cards: base {tbase / N:.4f}   per-family best in-sample {tin / N:.4f}   cross-validated {tcv / N:.4f}")
for label, cfg in (("F2,F4 x2 (t8)", {"F2": "t8 x2", "F4": "t8 x2"}), ("F2,F4 x2.5", {"F2": "t8 x2.5", "F4": "t8 x2.5"}),
                   ("F2,F4 x3", {"F2": "t8 x3", "F4": "t8 x3"}), ("F2,F4 x1.5", {"F2": "t8 x1.5", "F4": "t8 x1.5"}),
                   ("F2,F3,F4 x2", {"F2": "t8 x2", "F3": "t8 x2", "F4": "t8 x2"}),
                   ("F2,F4 t4 x2", {"F2": "t4 x2", "F4": "t4 x2"})):
    sc = [score[cfg.get(C.family(n), "t8 x1")][n] for n in sorted(names)]
    print(f"  fixed rule {label:16s} {np.mean(sc):.4f}   halves {np.mean(sc[0::2]):.3f} / {np.mean(sc[1::2]):.3f}")
