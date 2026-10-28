"""New-scale score of a per-family text-off config. Usage: score_config.py "F1:...;F2:...;F3:...;F4:..." [global variant]"""
import sys
import numpy as np
import t2_core as C
import t2_sim as T
import new_scale as N
truth = C.load_truth()
names = T.card_set(truth, "all")
spec = sys.argv[1]
glob = C.parse_variant(sys.argv[2]) if len(sys.argv) > 2 else {}
fam_v = {}
for chunk in spec.split(";"):
    f, _, s = chunk.partition(":")
    v = dict(glob)
    for part in s.split(","):
        k, _, val = part.strip().partition("=")
        if not k:
            continue
        try:
            v[k] = float(val)
        except ValueError:
            v[k] = val
    fam_v[f.strip()] = v
per = {}
for n in names:
    v = fam_v.get(C.family(n), glob)
    s, cells = C.m0x_draws(n, v or None)
    per[n] = min(N.score_new(n, s, C.truth_vector(truth, n, cells)), N.CLIP)
fams = {}
for n, x in per.items():
    fams.setdefault(C.family(n), []).append(x)
odd = np.mean([per[n] for n in names[0::2]]); even = np.mean([per[n] for n in names[1::2]])
print(f"mean {np.mean(list(per.values())):.4f}  halves {odd:.4f}/{even:.4f}  " + "  ".join(f"{f} {np.mean(v):.3f} (n={len(v)})" for f, v in sorted(fams.items())),
      " clipped", sum(1 for x in per.values() if x >= 8))
