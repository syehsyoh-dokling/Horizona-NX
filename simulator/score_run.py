"""Score a real run's forecasts (DIR/<card>/forecast.parquet) on the new scale, raw and calibrated,
per family, and the cards that moved most vs exact M0. Usage: score_run.py DIR [label]"""
import pathlib, sys
import numpy as np, pandas as pd
import t2_core as C, t2_sim as T, new_scale as N, calib
truth = C.load_truth(); names = T.card_set(truth, "all")
root = pathlib.Path(sys.argv[1]); label = sys.argv[2] if len(sys.argv) > 2 else root.name
rows = []
for n in names:
    f = root / n / "forecast.parquet"
    if not f.exists():
        from calibrate_live import RENAME
        old = {v: k for k, v in RENAME.items()}.get(n)
        if old: f = root / old / "forecast.parquet"
    base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells)
    m0 = min(calib.score_live(n, base, y), 8.0)
    if not f.exists():
        rows.append((n, C.family(n), None, None, m0)); continue
    df = pd.read_parquet(f); wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    S = np.column_stack([wide[(a, h)].to_numpy() for a, h in cells])
    rows.append((n, C.family(n), min(N.score_new(n, S, y), 8.0), min(calib.score_live(n, S, y), 8.0), m0))
have = [r for r in rows if r[2] is not None]
print(f"{label}: cards {len(have)}/{len(rows)} | raw new-scale {np.mean([r[2] for r in have]):.4f} | CALIBRATED {np.mean([r[3] for r in have]):.4f} | M0 calibrated {np.mean([r[4] for r in have]):.4f}")
for f in ("F1", "F2", "F3", "F4"):
    fr = [r for r in have if r[1] == f]
    print(f"  {f}: n={len(fr)} calibrated {np.mean([r[3] for r in fr]):.3f} (M0 {np.mean([r[4] for r in fr]):.3f})")
print("  biggest gains vs M0 (calibrated):")
for n, f, raw, cal, m0 in sorted(have, key=lambda r: r[3] - r[4])[:6]:
    print(f"    {n:38s} {m0:6.2f} -> {cal:5.2f}")
print("  biggest losses vs M0:")
for n, f, raw, cal, m0 in sorted(have, key=lambda r: r[4] - r[3])[:6]:
    print(f"    {n:38s} {m0:6.2f} -> {cal:5.2f}")
print("  cards at the clip (8.0):", sum(1 for r in have if r[3] >= 8.0))
