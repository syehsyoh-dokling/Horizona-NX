"""Census of readings.json (memory-forbidden narrative prompt): per card, max |direction_sd|,
max vol_mult, confidence; and whether these pre-outcome signals line up with the cards that the
calibrated newscale-free M0 scores badly (clipped or > 3). Usage: readings_census.py"""
import json, pathlib
import numpy as np
import t2_core as C, t2_sim as T, calib
truth = C.load_truth(); names = T.card_set(truth, "all"); fam = {n: C.family(n) for n in names}
R = json.load(open(pathlib.Path(__file__).resolve().parent / "readings.json"))
rows = []
for n in names:
    r = R.get(n, {}); A = r.get("assets", {})
    ds = max([abs(v.get("direction_sd", 0.0)) for v in A.values()] or [0.0])
    vm = max([v.get("vol_mult", 1.0) for v in A.values()] or [1.0])
    conf = sorted({v.get("confidence", "?") for v in A.values()})
    base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells)
    m0 = min(calib.score_live(n, base, y), 8.0)
    rows.append((n, fam[n], ds, vm, ",".join(conf), m0))
print(f"cards {len(rows)}; direction_sd != 0 on {sum(r[2] > 0 for r in rows)}; vol_mult > 1 on {sum(r[3] > 1 for r in rows)}; vol_mult >= 1.5 on {sum(r[3] >= 1.5 for r in rows)}")
m0s = np.array([r[5] for r in rows]); vm = np.array([r[3] for r in rows]); ds = np.array([r[2] for r in rows])
print(f"M0 calibrated mean {m0s.mean():.3f}; clipped(8) {np.sum(m0s >= 8)}; >3: {np.sum(m0s > 3)}")
for thr in (1.0, 1.2, 1.5, 2.0):
    sel = vm >= thr
    if sel.sum(): print(f"  vol_mult >= {thr}: n={sel.sum()}  M0 mean {m0s[sel].mean():.2f} (others {m0s[~sel].mean():.2f})  clipped {np.sum(m0s[sel] >= 8)}/{sel.sum()}")
sel = ds > 0
print(f"  direction given: n={sel.sum()}  M0 mean {m0s[sel].mean():.2f} (others {m0s[~sel].mean():.2f})")
print("worst 12 cards by M0 with their readings:")
for r in sorted(rows, key=lambda r: -r[5])[:12]:
    print(f"  {r[0]:38s} {r[1]} M0 {r[5]:5.2f}  dir_sd {r[2]:.2f} vol {r[3]:.2f} conf {r[4]}")
