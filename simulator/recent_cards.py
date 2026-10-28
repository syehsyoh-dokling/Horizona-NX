"""Does recall work on the most recent cards (the sealed Final is a 'later window' with panels ending
2024-12-31, i.e. within the House model's knowledge)? Per card with as-of >= 2023: calibrated
score of M0, newscale, and a recall run. Usage: recent_cards.py RUN_DIR"""
import pathlib, sys
import numpy as np, pandas as pd
import t2_core as C, t2_sim as T, calib
from calibrate_live import RENAME, NEWSCALE
truth = C.load_truth(); names = T.card_set(truth, "all")
root = pathlib.Path(sys.argv[1]); inv = {v: k for k, v in RENAME.items()}
rows = []
for n in names:
    card = C.card_of(n); asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    f = root / n / "forecast.parquet"
    if not f.exists() and inv.get(n): f = root / inv[n] / "forecast.parquet"
    base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells)
    S1, _ = C.m0x_draws(n, NEWSCALE[C.family(n)])
    df = pd.read_parquet(f); wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    S = np.column_stack([wide[(a, h)].to_numpy() for a, h in cells])
    rows.append((asof, n, C.family(n), min(calib.score_live(n, base, y, card), 8.0), min(calib.score_live(n, S1, y, card), 8.0), min(calib.score_live(n, S, y, card), 8.0)))
rows.sort()
print(f"{'asof':10s} {'card':38s} fam    M0  newscale  recall")
for asof, n, f, m0, ns, rc in rows:
    if asof >= pd.Timestamp("2022-01-01"):
        print(f"{asof.date()} {n:38s} {f}  {m0:5.2f}  {ns:5.2f}  {rc:5.2f}")
for lo in ("2016-01-01", "2020-01-01", "2022-01-01", "2023-06-01"):
    sub = [r for r in rows if r[0] >= pd.Timestamp(lo)]
    print(f"cards as-of >= {lo}: n={len(sub)}  M0 {np.mean([r[3] for r in sub]):.3f}  newscale {np.mean([r[4] for r in sub]):.3f}  recall {np.mean([r[5] for r in sub]):.3f}")
