"""Per-card divisor correction from the 13 Oct live board. The newscale upload is deterministic and
the simulator reproduces its draws byte-for-byte, so live/sim on that upload = (our divisor) /
(organizer divisor) per card - a single factor per card because a common sd scale moves the
marginal, tail and joint divisors alike. score_live(name, S, y) = score_new * factor."""
import json, pathlib
import numpy as np
import t2_core as C, t2_sim as T, new_scale as N
from calibrate_live import RENAME, NEWSCALE, live
H = pathlib.Path(__file__).resolve().parent
_F = H / "calib_factors.json"


def factors(recompute: bool = False) -> dict:
    if _F.exists() and not recompute:
        return json.load(open(_F))
    truth = C.load_truth(); out = {}
    for n in T.card_set(truth, "all"):
        base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells)
        s, _ = C.m0x_draws(n, NEWSCALE[C.family(n)])
        out[n] = live[RENAME.get(n, n)] / N.score_new(n, s, y)
    json.dump(out, open(_F, "w"), indent=1)
    return out


FACT = factors()


def score_live(name, S, y, card=None) -> float:
    return N.score_new(name, S, y, card) * FACT[name]


if __name__ == "__main__":
    truth = C.load_truth(); names = T.card_set(truth, "all")
    f = factors(True)
    v = np.array(list(f.values()))
    print(f"factors: median {np.median(v):.3f}  IQR {np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}  min {v.min():.3f} max {v.max():.3f}")
    for lab, fn in (("newscale (must equal live 1.9191)", lambda n: C.m0x_draws(n, NEWSCALE[C.family(n)])[0]),
                    ("exact M0 (organizer reference row)", lambda n: C.org_m0_draws(n)[0]),
                    ("drift05 old best", lambda n: C.m0x_draws(n, {"t_nu": 8.0, "drift": 0.5})[0])):
        per = []
        for n in names:
            _, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells)
            per.append(min(score_live(n, fn(n), y), 8.0))
        print(f"{lab:38s} {np.mean(per):.4f}")
