import t2_core as C
import t2_sim as T
import numpy as np

truth = C.load_truth()
names = T.card_set(truth, "all")

for f4_drift in ("t1", "zero", 0.5, 0.25):
    scores = {}
    for n in names:
        base, cells = T.base_of(n)
        y = C.truth_vector(truth, n, cells)
        fam = C.family(n)
        if fam == "F4":
            variant = {"t_nu": 8.0, "drift": f4_drift}
        else:
            variant = {"t_nu": 8.0, "drift": 0.5}
        S, _ = C.m0x_draws(n, variant)
        scores[n] = min(C.card_score(C.card_of(n), base, S, y), 4.0)
    
    mean_all = np.mean(list(scores.values()))
    fams = {f: np.mean([scores[n] for n in names if C.family(n) == f]) for f in T.FAMS}
    print(f"F4_drift={str(f4_drift):<6} -> Overall: {mean_all:.4f} | F1: {fams['F1']:.4f}, F2: {fams['F2']:.4f}, F3: {fams['F3']:.4f}, F4: {fams['F4']:.4f}")