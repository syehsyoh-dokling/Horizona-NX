"""Shared core of the local Track 2 leaderboard simulator.

Scoring = the organizers' documented scorer (qfbench2_track_forecasting/scoring.py, origin/main):
  components: qfbench2_common.scoring.crps.crps_marginal, variogram_score(p=0.5), and this track's
              pinball tail (tail.py, the default tail metric; copied verbatim below)
  per card  : weights . (component / M0's component), with a non-positive M0 component stored as
              1.0 (M0-BASELINE.md section 5), single-cell weights renormalised to (0.714, 0, 0.286)
  primary   : mean over cards of min(card score, 4.0)
M0 = our agent's forecast_core/m0_exact.py (live: 1.0000, 69/71 cards exactly 1).
"""
from __future__ import annotations

import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps

HERE = pathlib.Path(__file__).resolve().parent
UNITS = HERE / "repo" / "units"
AGENT = pathlib.Path("/mnt/c/Users/Saifuddin/Documents/agenthon-t1/track2_agent")
if not AGENT.is_dir():
    AGENT = pathlib.Path("C:/Users/Saifuddin/Documents/agenthon-t1/track2_agent")
sys.path.insert(0, str(AGENT))
from forecast_core import m0 as OLD  # noqa: E402  the vendored reference CLI walk (m0b upload)
from forecast_core import m0_exact as M  # noqa: E402  exact M0 (+ variants)

CLIP = 4.0
LEVELS = (0.01, 0.05, 0.95, 0.99)


# ------------------------------------------------------------------ cards
def unit_names(split: str | None = None) -> list[str]:
    out = []
    for u in sorted(UNITS.iterdir()):
        if not (u / "card.toml").is_file() or "EXAMPLE" in u.name:
            continue
        if split is None or card_of(u.name)["task"].get("split") == split:
            out.append(u.name)
    return out


_CARDS: dict[str, dict] = {}


def card_of(name: str) -> dict:
    if name not in _CARDS:
        _CARDS[name] = tomllib.loads((UNITS / name / "card.toml").read_text(encoding="utf-8"))
    return _CARDS[name]


def family(name: str) -> str:
    return name.split("-")[1]


# ------------------------------------------------------------------ scorer
def tail_pinball(samples, y, levels=LEVELS) -> float:
    """qfbench2_track_forecasting.tail.tail_pinball, verbatim."""
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = 0.0
    for a in levels:
        q = np.quantile(s, a, axis=0)
        loss = np.where(y >= q, a * (y - q), (1.0 - a) * (q - y))
        total += float(np.mean(loss))
    return total / len(tuple(levels))


def components(samples, y, card, tail: str = "pinball") -> np.ndarray:
    params = card.get("scoring", {}).get("params", {})
    levels = tuple(params.get("tail_levels", LEVELS))
    marg = crps.crps_marginal(samples, y)
    jnt = crps.variogram_score(samples, y, p=0.5)
    if tail == "pinball":
        tl = tail_pinball(samples, y, levels)
    else:
        tl = float(crps.tail_penalty(samples, y, levels=levels))
    return np.array([marg, jnt, tl], dtype=float)


def weights(card, n_cells: int) -> np.ndarray:
    w = card.get("scoring", {}).get("params", {}).get(
        "weights", {"marginal": 0.5, "joint": 0.3, "tail": 0.2})
    w = np.array([float(w["marginal"]), float(w["joint"]), float(w["tail"])])
    if n_cells == 1:
        live = w[0] + w[2]
        w = np.array([w[0] / live, 0.0, w[2] / live])
    return w


def card_score(card, base, S, y, mode: str = "per_component", tail: str = "pinball") -> float:
    """Leaderboard value of forecast S on a card whose M0 draws are `base` (unclipped)."""
    cb, cv = components(base, y, card, tail), components(S, y, card, tail)
    w = weights(card, base.shape[1])
    if mode == "per_component":
        r = np.where(cb > 0, cv / np.where(cb > 0, cb, 1.0), cv)   # scale <= 0 is stored as 1.0
        return float(w @ r)
    return float((w @ cv) / (w @ cb))                               # composite ratio (hypothesis)


# ------------------------------------------------------------------ engines
def m0x_draws(name: str, variant: dict | None = None):
    """Exact M0 (variant None/{}) or a variant of it: (samples [500, d], cells)."""
    card = card_of(name)
    s, cells, _info = M.draws(card, UNITS / name, variant=variant or None)
    return s, cells


# Cards whose sealed M0 cell order is the card's own [targets] list order, not sorted (M0-BASELINE.md
# 3.9: "on the remaining three it is the card's own [targets] list order"). Identified from the live
# exact-M0 upload: every validation card scored 1.0000 except these two (0.9985, 0.9994), and both
# numbers are reproduced exactly by scoring our sorted-order draws against a list-order M0.
LIST_ORDER_CARDS = {"t2-F3-conundrum-joint-2005", "t2-F4-quant-crowding-2007"}


def org_m0_draws(name: str):
    """The ORGANIZERS' M0 draws (the denominator), columns in sorted (asset, horizon) `cells` order.
    Equal to m0_exact's draws except on LIST_ORDER_CARDS, where Z's columns follow the card order."""
    ours, cells = m0x_draws(name)
    if name not in LIST_ORDER_CARDS:
        return ours, cells
    import zlib
    card = card_of(name)
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    ttype = str(card["targets"].get("target_type", "level"))
    order = [(str(a), int(h)) for a in card["targets"]["asset_ids"] for h in card["targets"]["horizons"]]
    A = sorted({a for a, _ in order})
    hist = {a: M.history(UNITS / name, a, asof) for a in A}
    frame = pd.concat({a: M.steps(hist[a], ttype) for a in A}, axis=1, join="inner").dropna()
    X = frame[A].to_numpy(float)
    mu, Sigma = X.mean(axis=0), np.atleast_2d(np.cov(X, rowvar=False))
    ai = {a: i for i, a in enumerate(A)}
    anchor = {a: (0.0 if ttype == "log_return" else float(hist[a].iloc[-1])) for a in A}
    sc = {(a, h): M.panel_steps(hist[a], h, asof) for a, h in order}
    d = len(order)
    mean = np.array([anchor[a] + sc[(a, h)] * mu[ai[a]] for a, h in order])
    cov = np.array([[min(sc[(a, h)], sc[(b, g)]) * Sigma[ai[a], ai[b]] for (b, g) in order]
                    for (a, h) in order]) + 1e-10 * np.eye(d)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    Z = np.random.default_rng(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF).standard_normal((500, d))
    base_o = mean + Z @ L.T
    col = {c: j for j, c in enumerate(order)}
    return np.column_stack([base_o[:, col[c]] for c in cells]), cells


def m0b_draws(name: str, cells, window: dict | None = None):
    """The live m0b upload: reference CLI walk, seed 0, 500 draws, scale from the last 504 changes
    on F1/F3 (T2_M0_FAMILY_CONFIG=F1:window=504;F3:window=504). Columns in `cells` order."""
    card = card_of(name)
    window = {"F1": 504, "F3": 504} if window is None else window
    assets = list(card["targets"]["asset_ids"])
    hs = [int(h) for h in card["targets"]["horizons"]]
    panels = OLD._read_panels(UNITS / name)
    s, _ = OLD._draw(panels, assets, hs, card["provenance"]["data_cutoff"], 500, 0,
                     window.get(family(name)))
    d = {(a, h): s[:, ai, hi] for ai, a in enumerate(assets) for hi, h in enumerate(hs)}
    return np.column_stack([d[c] for c in cells])


def parse_variant(text: str) -> dict:
    """'{}' / JSON / the agent's T2_M0_VARIANT syntax 't_nu=8,drift=0.5,ewma=0.97:0.5,widen=0.95'."""
    text = (text or "").strip()
    if not text or text in ("{}", "m0", "M0"):
        return {}
    if text.startswith("{"):
        return json.loads(text)
    out: dict = {}
    for part in text.split(","):
        key, _, val = part.strip().partition("=")
        if not key or not val:
            continue
        try:
            out[key] = [float(x) for x in val.split(":")] if ":" in val else float(val)
        except ValueError:
            out[key] = val
    return out


# ------------------------------------------------------------------ truth
def load_truth(path: pathlib.Path | None = None) -> dict:
    return json.loads((path or HERE / "truth.json").read_text())


def truth_vector(truth: dict, name: str, cells) -> np.ndarray | None:
    t = truth.get(name)
    if not t:
        return None
    by = {(c["asset"], int(c["horizon"])): c["value"] for c in t["cells"]}
    try:
        y = np.array([by[(a, int(h))] for a, h in cells], dtype=float)
    except KeyError:
        return None
    return y if np.all(np.isfinite(y)) else None


def panel_series(name: str, asset: str) -> pd.Series:
    return M.full_series(UNITS / name, asset)
