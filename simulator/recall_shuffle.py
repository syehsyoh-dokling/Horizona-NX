"""Final-phase downside test for a recall run: the sealed Final cards resolve in the future, so the
model's recalled centres carry no information there. Emulate that by shuffling the run's centre
moves (centre - M0 mean, in M0-sd units) across cards within the same family, then score the hedge
centre = M0 + a*(move) with width w, on the calibrated simulator. Reports real (memory) vs shuffled
(no information) for each (a, w), per family and overall. Usage: recall_shuffle.py RUN_DIR"""
import itertools, pathlib, sys
import numpy as np, pandas as pd
import t2_core as C, t2_sim as T, calib
from calibrate_live import RENAME
truth = C.load_truth(); names = T.card_set(truth, "all"); fam = {n: C.family(n) for n in names}
root = pathlib.Path(sys.argv[1]); inv = {v: k for k, v in RENAME.items()}
cache = {}
for n in names:
    f = root / n / "forecast.parquet"
    if not f.exists() and inv.get(n): f = root / inv[n] / "forecast.parquet"
    if not f.exists(): continue
    base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells); card = C.card_of(n)
    S0, _ = C.m0x_draws(n, None); mean = S0.mean(axis=0); sd = S0.std(axis=0) + 1e-12
    df = pd.read_parquet(f); wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    S = np.column_stack([wide[(a, h)].to_numpy() for a, h in cells]); c = S.mean(axis=0)
    cache[n] = (S0, mean, sd, (c - mean) / sd, y, card)
have = [n for n in names if n in cache]
print(f"cards with forecasts: {len(have)}/{len(names)}")
def score(moves, a, w):
    per = {}
    for n in have:
        S0, mean, sd, _, y, card = cache[n]
        mv = moves[n]
        k = min(len(mv), len(sd))
        centre = mean.copy(); centre[:k] = mean[:k] + a * mv[:k] * sd[:k]
        per[n] = min(calib.score_live(n, centre + (S0 - mean) * w, y, card), 8.0)
    return per
real = {n: cache[n][3] for n in have}
rng = np.random.default_rng(0)
def shuffled(seed):
    r = np.random.default_rng(seed); out = {}
    for f in ("F1", "F2", "F3", "F4"):
        fn = [n for n in have if fam[n] == f]; perm = r.permutation(len(fn))
        for i, n in enumerate(fn):
            src = cache[fn[perm[i]]][3]
            out[n] = src if len(src) == len(cache[n][3]) else np.resize(src, len(cache[n][3]))
    return out
m0 = score({n: np.zeros_like(real[n]) for n in have}, 0.0, 1.0)
print(f"M0 calibrated (a=0): {np.mean(list(m0.values())):.4f}")
print(f"{'a':>4} {'w':>4} | {'real(memory)':>12} | {'shuffled mean':>13} {'min':>6} {'max':>6} | clipped real/shuf")
for a, w in itertools.product((0.25, 0.5, 0.75, 1.0), (1.0, 1.5, 2.0)):
    pr = score(real, a, w); rs = [np.mean(list(score(shuffled(s), a, w).values())) for s in range(5)]
    print(f"{a:4.2f} {w:4.1f} | {np.mean(list(pr.values())):12.4f} | {np.mean(rs):13.4f} {min(rs):6.3f} {max(rs):6.3f} | {sum(v >= 8 for v in pr.values())}/{sum(v >= 8 for v in score(shuffled(0), a, w).values())}")
