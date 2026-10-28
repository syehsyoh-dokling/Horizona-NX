"""Where is the gap to the oracle? (dev analysis; base = t8 + drift x0.5 with F4 drift |t|>=1)

Per card: the realised move in units of the forecast's own sd (z = |truth - centre| / sd), the
score now, and the score if the centre sat at the truth with the same width (oracle centre).
Then: by family, by size of the move, by score component, and the cards that hold the gap.
"""
import numpy as np

import t2_core as C
import t2_sim as T

truth = C.load_truth()
names = T.card_set(truth, "all")
rows = []
for n in names:
    base, cells = T.base_of(n)
    variant = {"t_nu": 8.0, "drift": "t1"} if C.family(n) == "F4" else {"t_nu": 8.0, "drift": 0.5}
    S, _ = C.m0x_draws(n, variant)
    y = C.truth_vector(truth, n, cells)
    card = C.card_of(n)
    c, sd = S.mean(axis=0), S.std(axis=0)
    z = float(np.mean(np.abs(y - c) / sd))
    now = min(C.card_score(card, base, S, y), C.CLIP)
    orc = min(C.card_score(card, base, S + (y - c), y), C.CLIP)
    half = min(C.card_score(card, base, S + 0.5 * (y - c), y), C.CLIP)
    cb, cv = C.components(base, y, card), C.components(S, y, card)
    w = C.weights(card, base.shape[1])
    comp = w * np.where(cb > 0, cv / np.where(cb > 0, cb, 1.0), cv)
    rows.append((n, C.family(n), z, now, half, orc, comp, len(cells)))

N = len(rows)
print(f"all {N} cards: now {np.mean([r[3] for r in rows]):.4f}  centre half-way {np.mean([r[4] for r in rows]):.4f}"
      f"  centre at truth {np.mean([r[5] for r in rows]):.4f}")
print("\nBY FAMILY        n   mean|z|  now    half   oracle   gap now-oracle (share of total gap)")
total_gap = sum(r[3] - r[5] for r in rows)
for f in T.FAMS:
    fr = [r for r in rows if r[1] == f]
    gap = sum(r[3] - r[5] for r in fr)
    print(f"  {f}            {len(fr):2d}   {np.mean([r[2] for r in fr]):.2f}   {np.mean([r[3] for r in fr]):.3f}  "
          f"{np.mean([r[4] for r in fr]):.3f}  {np.mean([r[5] for r in fr]):.3f}    {gap / N:.3f} ({gap / total_gap:.0%})")

print("\nBY SIZE OF THE REALISED MOVE (|z| = move / forecast sd)")
for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 9)):
    br = [r for r in rows if lo <= r[2] < hi]
    if br:
        gap = sum(r[3] - r[5] for r in br)
        print(f"  |z| {lo:.1f}-{hi:.1f}   n={len(br):2d}  now {np.mean([r[3] for r in br]):.3f}  oracle "
              f"{np.mean([r[5] for r in br]):.3f}  share of gap {gap / total_gap:.0%}")

comp = np.mean([r[6] for r in rows], axis=0)
print(f"\nSCORE NOW BY COMPONENT (weighted): marginal CRPS {comp[0]:.3f}  variogram {comp[1]:.3f}  tail {comp[2]:.3f}")

print("\nTHE 15 CARDS HOLDING THE MOST GAP (now - oracle)")
for r in sorted(rows, key=lambda r: r[5] - r[3])[:15]:
    print(f"  {r[0]:38s} |z|={r[2]:.2f}  now {r[3]:.3f}  half {r[4]:.3f}  oracle {r[5]:.3f}  cells {r[7]}")
top = sorted(rows, key=lambda r: r[5] - r[3])[:15]
print(f"  these 15 hold {sum(r[3] - r[5] for r in top) / total_gap:.0%} of the gap")
small = [r for r in rows if r[3] - r[5] < 0.15]
print(f"\ncards where a perfect centre would gain less than 0.15: {len(small)} of {N}")
