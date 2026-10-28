"""T2 grid of statistical configurations (text off, 71 scored cards): global variants and the one
family override that was stable in cross-validation (F4: drift kept only where |t| >= 1).
Prints mean, family means, both split halves, and the paired difference vs the live base (d05).
"""
import json

import numpy as np

import t2_core as C
import t2_sim as T

G = {
    "t8 (no drift change)": {"t_nu": 8.0},
    "d05 (live base)": {"t_nu": 8.0, "drift": 0.5},
    "d075": {"t_nu": 8.0, "drift": 0.75},
    "d06": {"t_nu": 8.0, "drift": 0.6},
    "d04": {"t_nu": 8.0, "drift": 0.4},
    "d05 t5": {"t_nu": 5.0, "drift": 0.5},
    "d05 t12": {"t_nu": 12.0, "drift": 0.5},
    "d05 gauss": {"drift": 0.5},
    "d05 ewma.97/.5": {"t_nu": 8.0, "drift": 0.5, "ewma": [0.97, 0.5]},
    "d05 ewma.97/.25": {"t_nu": 8.0, "drift": 0.5, "ewma": [0.97, 0.25]},
    "d05 ewma.94/.25": {"t_nu": 8.0, "drift": 0.5, "ewma": [0.94, 0.25]},
    "d05 widen1.05": {"t_nu": 8.0, "drift": 0.5, "widen": 1.05},
    "d05 widen1.10": {"t_nu": 8.0, "drift": 0.5, "widen": 1.10},
    "d05 widen0.95": {"t_nu": 8.0, "drift": 0.5, "widen": 0.95},
    "d075 ewma.97/.25": {"t_nu": 8.0, "drift": 0.75, "ewma": [0.97, 0.25]},
    "d075 widen1.05": {"t_nu": 8.0, "drift": 0.75, "widen": 1.05},
    "dt1": {"t_nu": 8.0, "drift": "t1"},
    "dt1 widen1.05": {"t_nu": 8.0, "drift": "t1", "widen": 1.05},
}
truth = C.load_truth()
names = T.card_set(truth, "all")
S = {k: {n: min(v, C.CLIP) for n, v in T.simulate(var, "m0x", names, truth).items()} for k, var in G.items()}
json.dump(S, open(T.HERE / "grid_combo_scores.json", "w"), indent=1)


def mix(default, overrides):
    return {n: S[overrides.get(C.family(n), default)][n] for n in names}


ROWS = {k: S[k] for k in G}
ROWS["d05 + F4:dt1"] = mix("d05 (live base)", {"F4": "dt1"})
ROWS["d075 + F4:dt1"] = mix("d075", {"F4": "dt1"})
ROWS["d05 + F4:t8"] = mix("d05 (live base)", {"F4": "t8 (no drift change)"})
ROWS["d05 + F4:dt1 widen1.05"] = mix("d05 (live base)", {"F4": "dt1 widen1.05"})
ROWS["d05 widen1.05 + F4:dt1"] = mix("d05 widen1.05", {"F4": "dt1"})
ROWS["d05 ewma.97/.25 + F4:dt1"] = mix("d05 ewma.97/.25", {"F4": "dt1"})

base = ROWS["d05 (live base)"]
srt = sorted(names)
a, b = srt[0::2], srt[1::2]
print(f"{'configuration':28s}  mean    F1     F2     F3     F4    half-A half-B  vs d05 (better/worse, t)")
for k, sc in sorted(ROWS.items(), key=lambda kv: np.mean(list(kv[1].values()))):
    fam = [np.mean([sc[n] for n in names if C.family(n) == f]) for f in T.FAMS]
    d = np.array([sc[n] - base[n] for n in names])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
    print(f"{k:28s} {np.mean(list(sc.values())):.4f}  " + "  ".join(f"{v:.3f}" for v in fam)
          + f"   {np.mean([sc[n] for n in a]):.3f}  {np.mean([sc[n] for n in b]):.3f}"
          + f"   {int((d < 0).sum()):2d}/{int((d > 0).sum()):2d}  t={t:+.2f}")
