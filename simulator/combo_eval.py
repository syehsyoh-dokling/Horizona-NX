"""Per-family choice among: base, document direction, model recall (71 cards, cross-validated)."""
import json

import numpy as np

import t2_core as C
import t2_sim as T

truth = C.load_truth()
names = T.card_set(truth, "all")
std = json.load(open(T.HERE / "readings.json"))
rec = json.load(open(T.HERE / "recall.json"))
BASE = {"t_nu": 8.0, "drift": 0.5}


def fnum(e, k, default):
    try:
        return float(e.get(k, default) or default)
    except (TypeError, ValueError, AttributeError):
        return default


def forecast(n, S, cells, mode):
    c, sd = S.mean(axis=0), S.std(axis=0)
    out = S.copy()
    kind, w = mode
    for j, (a, h) in enumerate(cells):
        shift = 0.0
        if kind in ("dir", "dir+rec"):
            e = ((std.get(n) or {}).get("assets") or {}).get(a) or {}
            shift += max(-1.0, min(1.0, fnum(e, "direction_sd", 0.0))) * sd[j] * (w if kind == "dir" else 1.0)
        if kind in ("rec", "dir+rec"):
            r = rec.get(n) or {}
            e = (r.get("cells") or {}).get(f"{a}@{h}") or {}
            pt = fnum(e, "point", None)
            if pt is not None and r.get("recognised") is True and abs(pt - c[j]) <= 8 * sd[j]:
                shift = w * (pt - c[j])         # recall replaces the document direction
        out[:, j] = S[:, j] + shift
    return out


MODES = {"base": ("base", 0), "dir1": ("dir", 1.0), "rec25": ("rec", 0.25), "rec50": ("rec", 0.5),
         "rec75": ("rec", 0.75), "rec100": ("rec", 1.0), "dir+rec50": ("dir+rec", 0.5),
         "dir+rec75": ("dir+rec", 0.75)}
score = {m: {} for m in MODES}
for n in names:
    base, cells = T.base_of(n)
    S, _ = C.m0x_draws(n, BASE)
    y = C.truth_vector(truth, n, cells)
    for m, mode in MODES.items():
        score[m][n] = min(C.card_score(C.card_of(n), base, forecast(n, S, cells, mode), y), C.CLIP)

print("mode        " + "  ".join(f"{f:>6s}" for f in T.FAMS) + "     all")
for m in MODES:
    fam = [np.mean([score[m][n] for n in names if C.family(n) == f]) for f in T.FAMS]
    print(f"{m:11s} " + "  ".join(f"{v:6.3f}" for v in fam) + f"  {np.mean(list(score[m].values())):6.4f}")

tin = tcv = 0.0
print("\nfamily  n  in-sample best      cross-validated (pick on one half, score the other)")
for f in T.FAMS:
    cards = sorted(n for n in names if C.family(n) == f)
    a, b = cards[0::2], cards[1::2]
    mean = lambda m, cs: float(np.mean([score[m][n] for n in cs]))  # noqa: E731
    best = min(MODES, key=lambda m: mean(m, cards))
    pa, pb = min(MODES, key=lambda m: mean(m, a)), min(MODES, key=lambda m: mean(m, b))
    cv = (mean(pa, b) * len(b) + mean(pb, a) * len(a)) / len(cards)
    print(f"{f}     {len(cards):2d} {best:10s} {mean(best, cards):.3f}    {cv:.3f}  (picks {pa} / {pb})")
    tin += mean(best, cards) * len(cards); tcv += cv * len(cards)
print(f"\nper-family: in-sample {tin / len(names):.4f}, cross-validated {tcv / len(names):.4f}")
fixed = {"F1": "dir1", "F2": "rec100", "F3": "dir1", "F4": "rec100"}
print("fixed rule (recall on F2/F4, document direction on F1/F3):",
      f"{np.mean([score[fixed[C.family(n)]][n] for n in names]):.4f}")
fixed2 = {"F1": "dir1", "F2": "rec75", "F3": "dir1", "F4": "rec75"}
print("same with recall weight 0.75:", f"{np.mean([score[fixed2[C.family(n)]][n] for n in names]):.4f}")
