import t2_core as C
import t2_sim as T
import numpy as np

truth = C.load_truth()
names = T.card_set(truth, "all")

# Candidate 1: Global d05 + F4:dt1 (The 0.9585 base)
# Candidate 2: Global d05 + F4:dt1 + F2:ewma
# Candidate 3: Global d05 + F4:dt1 + F1:t5
for label, fam_cfg in [
    ("Base (d05 + F4:dt1)", {"F1": {"t_nu": 8.0, "drift": 0.5}, "F2": {"t_nu": 8.0, "drift": 0.5}, "F3": {"t_nu": 8.0, "drift": 0.5}, "F4": {"t_nu": 8.0, "drift": "t1"}}),
    ("+ F2:ewma", {"F1": {"t_nu": 8.0, "drift": 0.5}, "F2": {"t_nu": 8.0, "drift": 0.5, "ewma": [0.97, 0.5]}, "F3": {"t_nu": 8.0, "drift": 0.5}, "F4": {"t_nu": 8.0, "drift": "t1"}}),
    ("+ F1:t5", {"F1": {"t_nu": 5.0, "drift": 0.5}, "F2": {"t_nu": 8.0, "drift": 0.5}, "F3": {"t_nu": 8.0, "drift": 0.5}, "F4": {"t_nu": 8.0, "drift": "t1"}}),
    ("+ F2:ewma + F1:t5", {"F1": {"t_nu": 5.0, "drift": 0.5}, "F2": {"t_nu": 8.0, "drift": 0.5, "ewma": [0.97, 0.5]}, "F3": {"t_nu": 8.0, "drift": 0.5}, "F4": {"t_nu": 8.0, "drift": "t1"}}),
]:
    scores = {}
    for n in names:
        base, cells = T.base_of(n)
        y = C.truth_vector(truth, n, cells)
        fam = C.family(n)
        variant = fam_cfg[fam]
        S, _ = C.m0x_draws(n, variant)
        scores[n] = min(C.card_score(C.card_of(n), base, S, y), 4.0)
    
    mean_all = np.mean(list(scores.values()))
    fams = {f: np.mean([scores[n] for n in names if C.family(n) == f]) for f in T.FAMS}
    print(f"{label:<22} -> Overall: {mean_all:.4f} | F1: {fams['F1']:.4f}, F2: {fams['F2']:.4f}, F3: {fams['F3']:.4f}, F4: {fams['F4']:.4f}")