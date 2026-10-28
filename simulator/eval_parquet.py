"""Score real agent outputs (a directory of <card>/forecast.parquet + run.log) in the simulator.

Usage: eval_parquet.py DIR    -> per-family means, split halves, and how each card's centre moved.
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

import t2_core as C
import t2_sim as T

root = pathlib.Path(sys.argv[1])
truth = C.load_truth()
names = T.card_set(truth, "all")
rows = []
for n in names:
    f = root / n / "forecast.parquet"
    if not f.exists():
        rows.append((n, None, "missing"))
        continue
    base, cells = T.base_of(n)
    df = pd.read_parquet(f)
    wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    S = np.column_stack([wide[(a, h)].to_numpy() for a, h in cells])
    y = C.truth_vector(truth, n, cells)
    score = min(C.card_score(C.card_of(n), base, S, y), C.CLIP)
    log = (root / n / "run.log").read_text(errors="replace") if (root / n / "run.log").exists() else ""
    m = re.search(r'"centre_moves": (\{.*?\})', log)
    moves = set(json.loads(m.group(1)).values()) if m else set()
    kind = "recall" if "recall" in moves else ("direction" if "direction" in moves else "none")
    note = re.search(r'recall: ([^"]*)', log)
    rows.append((n, score, kind + (" | " + note.group(1)[:50] if note and kind != "recall" else "")))

ok = [(n, s, k) for n, s, k in rows if s is not None]
print(f"cards scored {len(ok)}/{len(names)}  primary {np.mean([s for _, s, _ in ok]):.4f}")
for f in T.FAMS:
    fam = [(n, s, k) for n, s, k in ok if C.family(n) == f]
    kinds = {}
    for _n, s, k in fam:
        kinds.setdefault(k.split(" |")[0], []).append(s)
    print(f"  {f} n={len(fam):2d} mean {np.mean([s for _, s, _ in fam]):.4f}   "
          + "  ".join(f"{k}: {len(v)} cards {np.mean(v):.3f}" for k, v in sorted(kinds.items())))
srt = sorted(ok)
print("  halves", f"{np.mean([s for _, s, _ in srt[0::2]]):.4f}", f"{np.mean([s for _, s, _ in srt[1::2]]):.4f}")
if "--cards" in sys.argv:
    for n, s, k in ok:
        print(f"    {n:40s} {s:.4f} {k}")
