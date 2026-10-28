"""Local rebuild of the Track 2 scorer for the short-horizon practice set.

Reproduces qfbench2_track_forecasting.scoring._composite / _score (raw, "raw_unrankable") and
adds the M0 expected-error divisors of docs/M0-BASELINE.md section 5 so that cards in different
units (KRW ~1000/USD, EUR ~1.1/USD, bills in % p.a.) can be averaged on the leaderboard's scale.

    python rawscore.py score   <unit_dir> <forecast.parquet> <realized.parquet>      -> JSON
    python rawscore.py divisor <unit_dir>                                             -> JSON

Importable: composite(samples, y, card) -> dict(marginal, joint, tail, composite, weights);
            divisors(mean, cov) -> (marg, joint, tail); m0_mean_cov(card, panels_dir).
Depends on qfbench2_common (the organizers' toolkit, the same functions the official scorer
calls) and on the team agent's forecast_core.m0_exact for M0's mean and covariance.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps
from scipy.special import gamma as Gamma
from scipy.special import hyp1f1
from scipy.stats import norm

HERE = pathlib.Path(__file__).resolve().parent
AGENT = HERE.parent / "repo" / "agent"
sys.path.insert(0, str(AGENT))
from forecast_core import m0_exact as M  # noqa: E402

LEVELS = (0.01, 0.05, 0.95, 0.99)
PHI_MEAN = float(np.mean([norm.pdf(norm.ppf(t)) for t in LEVELS]))  # 0.0648939
CLIP = 8.0


def tail_pinball(samples, y, levels=LEVELS) -> float:
    """qfbench2_track_forecasting.tail.tail_pinball, same arithmetic."""
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = 0.0
    for a in levels:
        q = np.quantile(s, a, axis=0)
        loss = np.where(y >= q, a * (y - q), (1.0 - a) * (q - y))
        total += float(np.mean(loss))
    return total / len(tuple(levels))


def weights(card: dict, n_cells: int) -> np.ndarray:
    w = card.get("scoring", {}).get("params", {}).get(
        "weights", {"marginal": 0.5, "joint": 0.3, "tail": 0.2})
    w = np.array([float(w["marginal"]), float(w["joint"]), float(w["tail"])])
    if n_cells == 1:  # single-cell renormalisation (scoring.py _score)
        live = w[0] + w[2]
        w = np.array([w[0] / live, 0.0, w[2] / live])
    return w


def cells_of(card: dict) -> list[tuple[str, int]]:
    tg = card["targets"]
    assets = sorted(str(a) for a in tg["asset_ids"])
    horizons = sorted(int(h) for h in tg["horizons"])
    return [(a, h) for a in assets for h in horizons]


def matrix_from_parquet(path: pathlib.Path, cells) -> np.ndarray:
    """forecast.parquet (draw, asset, horizon, value) -> [n_draws, d] in `cells` order."""
    df = pd.read_parquet(path)
    piv = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    cols = [(a, h) for a, h in cells]
    piv.columns = [(str(a), int(h)) for a, h in piv.columns]
    return piv[cols].to_numpy(dtype=float)


def realized_vector(path: pathlib.Path, cells) -> np.ndarray:
    df = pd.read_parquet(path)
    look = {(str(r.asset), int(r.horizon)): float(r.value) for r in df.itertuples()}
    return np.array([look[c] for c in cells], dtype=float)


def components(samples, y, card) -> np.ndarray:
    params = card.get("scoring", {}).get("params", {})
    levels = tuple(params.get("tail_levels", LEVELS))
    marg = float(crps.crps_marginal(samples, y))
    jnt = float(crps.variogram_score(samples, y, p=0.5))
    tl = tail_pinball(samples, y, levels)
    return np.array([marg, jnt, tl], dtype=float)


def composite(samples, y, card) -> dict:
    cv = components(samples, y, card)
    w = weights(card, samples.shape[1])
    return {"marginal": cv[0], "joint": cv[1], "tail": cv[2],
            "composite": float(w @ cv), "weights": w.tolist()}


# ---------------------------------------------------------------- M0 expected error (section 5)
def m0_mean_cov(card: dict, panels_dir: pathlib.Path):
    tgt = card["targets"]
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    ttype = str(tgt.get("target_type", "level"))
    assets = sorted(str(a) for a in tgt["asset_ids"])
    horizons = sorted(int(h) for h in tgt["horizons"])
    hist = {a: M.history(panels_dir, a, asof) for a in assets}
    frame = pd.concat({a: M.steps(hist[a], ttype) for a in assets}, axis=1, join="inner").dropna()
    X = frame[assets].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    Sigma = np.atleast_2d(np.cov(X, rowvar=False))
    cells = [(a, h) for a in assets for h in horizons]
    ai = {a: i for i, a in enumerate(assets)}
    # Section 3.7 on a business-daily panel with the sealed target dates: "steps 1-5 keep the
    # declared horizon on every one of their cells" (the comparison is STRICT, ratio > 2; a 2-day
    # horizon from a Monday counts 1 step, ratio exactly 2.0, declared horizon kept). The divisor
    # therefore uses s = declared horizon. (The team's m0_exact.panel_steps used >= 2 - see REPORT.)
    sc = {(a, h): int(h) for a, h in cells}
    anchor = {a: (0.0 if ttype == "log_return" else float(hist[a].iloc[-1])) for a in assets}
    d = len(cells)
    mean = np.array([anchor[a] + sc[(a, h)] * mu[ai[a]] for a, h in cells])
    cov = np.empty((d, d))
    for i, (a, h) in enumerate(cells):
        for j, (b, g) in enumerate(cells):
            cov[i, j] = min(sc[(a, h)], sc[(b, g)]) * Sigma[ai[a], ai[b]]
    cov = cov + 1e-10 * np.eye(d) + 1e-9 * np.eye(d)
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
        joint = 1.0
    return marg, joint, tail


def scaled(comp: dict, div: tuple[float, float, float]) -> float:
    """Leaderboard-scale composite (unclipped): weights . (component / divisor)."""
    w = np.array(comp["weights"])
    cv = np.array([comp["marginal"], comp["joint"], comp["tail"]])
    return float(w @ (cv / np.array(div)))


def read_card(unit_dir: pathlib.Path) -> dict:
    return tomllib.loads((unit_dir / "card.toml").read_text(encoding="utf-8"))


def panels_dir_of(unit_dir: pathlib.Path) -> pathlib.Path:
    return unit_dir / "panels" if (unit_dir / "panels").is_dir() else unit_dir


def main(argv: list[str]) -> int:
    verb = argv[0]
    unit_dir = pathlib.Path(argv[1])
    card = read_card(unit_dir)
    cells = cells_of(card)
    if verb == "divisor":
        mean, cov, _ = m0_mean_cov(card, panels_dir_of(unit_dir))
        print(json.dumps({"card_id": card["task"]["id"], "divisors": divisors(mean, cov)}))
        return 0
    if verb == "score":
        S = matrix_from_parquet(pathlib.Path(argv[2]), cells)
        y = realized_vector(pathlib.Path(argv[3]), cells)
        comp = composite(S, y, card)
        mean, cov, _ = m0_mean_cov(card, panels_dir_of(unit_dir))
        div = divisors(mean, cov)
        out = {"card_id": card["task"]["id"], "normalization_mode": "raw_unrankable",
               "marginal_crps": comp["marginal"], "joint_variogram": comp["joint"],
               "tail_penalty": comp["tail"], "composite_score": comp["composite"],
               "n_draws": int(S.shape[0]), "cell_count": len(cells),
               "m0_divisors": div, "scaled_composite": scaled(comp, div)}
        print(json.dumps(out, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
