"""Track 2 scoring under the 6 Oct 2026 rule (docs/M0-BASELINE.md s5, CONCEPTS s13):
each component is divided by M0's EXPECTED error computed from M0's own normal distribution
(mean, cov) built from the card's inputs; clip and failure value 8.0; single-cell weights
(0.714, 0, 0.286). M0's own score is no longer 1.0 - it is its realized error / expected error.

new_scale(name) -> (marginal_div, joint_div_or_1.0, tail_div), exact formulas (no sampling).
score_new(name, S, y) -> composite on the new scale (unclipped). Usage as a script:
  new_scale.py [variant]   -> mean new-scale score of exact M0, m0b, and the variant, 71 cards.
"""
from __future__ import annotations
import math
import sys
import numpy as np
import pandas as pd
from scipy.special import hyp1f1, gamma as Gamma
from scipy.stats import norm
import t2_core as C
import t2_sim as T
from forecast_core import m0_exact as M

LEVELS = (0.01, 0.05, 0.95, 0.99)
PHI_MEAN = float(np.mean([norm.pdf(norm.ppf(t)) for t in LEVELS]))   # 0.0648939
CLIP = 8.0


def m0_mean_cov(name: str):
    """Exact M0's (mean, cov, cells) in the simulator's sorted cell order, no variant."""
    card = C.card_of(name)
    tgt = card["targets"]
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    ttype = str(tgt.get("target_type", "level"))
    assets = sorted(str(a) for a in tgt["asset_ids"])
    horizons = sorted(int(h) for h in tgt["horizons"])
    hist = {a: M.history(C.UNITS / name, a, asof) for a in assets}
    frame = pd.concat({a: M.steps(hist[a], ttype) for a in assets}, axis=1, join="inner").dropna()
    X = frame[assets].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    Sigma = np.atleast_2d(np.cov(X, rowvar=False))
    cells = [(a, h) for a in assets for h in horizons]
    ai = {a: i for i, a in enumerate(assets)}
    sc = {(a, h): M.panel_steps(hist[a], h, asof) for a, h in cells}
    anchor = {a: (0.0 if ttype == "log_return" else float(hist[a].iloc[-1])) for a in assets}
    d = len(cells)
    mean = np.array([anchor[a] + sc[(a, h)] * mu[ai[a]] for a, h in cells])
    cov = np.empty((d, d))
    for i, (a, h) in enumerate(cells):
        for j, (b, g) in enumerate(cells):
            cov[i, j] = min(sc[(a, h)], sc[(b, g)]) * Sigma[ai[a], ai[b]]
    cov = cov + 1e-10 * np.eye(d)
    return mean, cov, cells


def _e_abs(delta: float, s: float) -> float:
    if s <= 0:
        return abs(delta)
    return s * math.sqrt(2 / math.pi) * math.exp(-delta * delta / (2 * s * s)) + delta * (1 - 2 * norm.cdf(-delta / s))


def _e_sqrt_abs(delta: float, s: float) -> float:
    if s <= 0:
        return math.sqrt(abs(delta))
    return math.sqrt(s) * 2 ** 0.25 * Gamma(0.75) / math.sqrt(math.pi) * float(hyp1f1(-0.25, 0.5, -delta * delta / (2 * s * s)))


def divisors(mean: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    sd = np.sqrt(np.diag(cov))
    marg = float(np.mean(sd) / math.sqrt(math.pi))
    tail = float(PHI_MEAN * np.mean(sd))
    d = len(mean)
    joint = 0.0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            delta = float(mean[i] - mean[j])
            s2 = float(cov[i, i] + cov[j, j] - 2 * cov[i, j])
            s = math.sqrt(max(s2, 0.0))
            joint += _e_abs(delta, s) - _e_sqrt_abs(delta, s) ** 2
    if joint <= 0:
        joint = 1.0            # "a component that comes out zero is stored as 1.0"
    return marg, joint, tail


_CACHE: dict[str, tuple[float, float, float]] = {}


def new_scale(name: str) -> tuple[float, float, float]:
    if name not in _CACHE:
        mean, cov, _ = m0_mean_cov(name)
        _CACHE[name] = divisors(mean, cov)
    return _CACHE[name]


def score_new(name: str, S: np.ndarray, y: np.ndarray, card=None) -> float:
    card = card or C.card_of(name)
    cv = C.components(S, y, card, "pinball")
    div = np.array(new_scale(name))
    w = C.weights(card, S.shape[1])
    return float(w @ (cv / div))


def mean_over(names, fn) -> float:
    vals = [min(fn(n), CLIP) for n in names]
    return float(np.mean(vals))


if __name__ == "__main__":
    truth = C.load_truth()
    names = T.card_set(truth, "all")
    variant = C.parse_variant(sys.argv[1]) if len(sys.argv) > 1 else {"t_nu": 8.0, "drift": 0.5}

    def sc_exact(n):
        base, cells = C.org_m0_draws(n)
        return score_new(n, base, C.truth_vector(truth, n, cells))

    def sc_m0b(n):
        base, cells = C.org_m0_draws(n)
        return score_new(n, C.m0b_draws(n, cells), C.truth_vector(truth, n, cells))

    def sc_var(n):
        s, cells = C.m0x_draws(n, variant)
        return score_new(n, s, C.truth_vector(truth, n, cells))

    for label, fn in (("exact M0 (organizers' reference row)", sc_exact), ("m0b upload", sc_m0b), (f"variant {variant}", sc_var)):
        per = {n: min(fn(n), CLIP) for n in names}
        fams = {}
        for n, v in per.items():
            fams.setdefault(C.family(n), []).append(v)
        print(f"{label:45s} mean {np.mean(list(per.values())):.4f}  " + "  ".join(f"{f} {np.mean(v):.3f}" for f, v in sorted(fams.items())))
