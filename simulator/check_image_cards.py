"""Compare an image's forecast.parquet with the simulator's variant draws on the new scale.
Usage: check_image_cards.py OUT_ROOT "F1:...;F2:...;F3:...;F4:..." card..."""
import sys, pathlib
import numpy as np, pandas as pd
import t2_core as C, t2_sim as T, new_scale as N
truth = C.load_truth()
root = pathlib.Path(sys.argv[1]); spec = sys.argv[2]
fam_v = {}
for chunk in spec.split(";"):
    f, _, s = chunk.partition(":"); v = {}
    for part in s.split(","):
        k, _, val = part.strip().partition("=")
        if k:
            try: v[k] = float(val)
            except ValueError: v[k] = val
    fam_v[f.strip()] = v
for n in sys.argv[3:]:
    s_sim, cells = C.m0x_draws(n, fam_v.get(C.family(n)) or None)
    y = C.truth_vector(truth, n, cells)
    df = pd.read_parquet(root / n / "forecast.parquet")
    wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    S = np.column_stack([wide[(a, h)].to_numpy() for a, h in cells])
    print(f"{n:34s} sim {N.score_new(n, s_sim, y):.4f} image {N.score_new(n, S, y):.4f}  max|diff| draws {np.max(np.abs(np.sort(S, axis=0) - np.sort(s_sim, axis=0))):.3g}")
