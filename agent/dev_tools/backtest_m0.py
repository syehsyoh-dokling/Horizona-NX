"""Backtest variants of M0 against M0 itself - the leaderboard's own ratio, on historical origins.

For every public card (monthly-panel cards skipped: their realized cell needs the sealed target
date), origins are moved BACK inside the card's own history, so every realized value is a panel
row at or before the card's as-of and nothing sealed is touched. At each origin M0 and each variant
forecast the card's grid; both are scored with the organizers' composite components
(qfbench2_common crps_marginal / variogram_score, this track's pinball tail) and the variant's
per-card score is the leaderboard's ratio: weights . (variant component / M0 component), with the
single-cell weight renormalization and the 4.0 clip. M0 is 1.000 by construction.

Needs the organizer toolkit (qfbench2-common v2.4.4) and this track's package installed.

Usage: backtest_m0.py UNITS_DIR [--origins 10] [--gap 42] [--only NAME ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from forecast_core import m0_exact as M  # noqa: E402

from qfbench2_common.scoring import crps  # noqa: E402
from qfbench2_track_forecasting.scoring import card_joint_statistic  # noqa: E402
from qfbench2_track_forecasting.tail import DEFAULT_TAIL_METRIC, TAIL_METRICS  # noqa: E402

VARIANTS: dict[str, dict] = {
    "M0": {},
    "drift0": {"drift": "zero"},
    "drift.5": {"drift": 0.5},
    "drift_t2": {"drift": "t2"},
    "ewma94": {"ewma": (0.94, 1.0)},
    "ewma94b": {"ewma": (0.94, 0.5)},
    "ewma97b": {"ewma": (0.97, 0.5)},
    "t5": {"t_nu": 5},
    "t8": {"t_nu": 8},
    "wide1.1": {"widen": 1.1},
    "narrow0.9": {"widen": 0.9},
    "d0+ewma94b": {"drift": "zero", "ewma": (0.94, 0.5)},
    "dt2+ewma94b+t8": {"drift": "t2", "ewma": (0.94, 0.5), "t_nu": 8},
}


def components(samples, y, card):
    params = card.get("scoring", {}).get("params", {})
    levels = tuple(params.get("tail_levels", (0.01, 0.05, 0.95, 0.99)))
    joint = card_joint_statistic(card)
    marg = crps.crps_marginal(samples, y)
    jnt = (crps.energy_score(samples, y) if joint == "energy"
           else crps.variogram_score(samples, y, p=0.5))
    tail = TAIL_METRICS[str(params.get("tail_metric", DEFAULT_TAIL_METRIC))](samples, y, levels)
    return np.array([marg, jnt, tail], dtype=float)


def weights(card, cells):
    w = card.get("scoring", {}).get("params", {}).get("weights",
                                                      {"marginal": 0.5, "joint": 0.3, "tail": 0.2})
    w = np.array([float(w["marginal"]), float(w["joint"]), float(w["tail"])])
    if cells == 1:
        live = w[0] + w[2]
        w = np.array([w[0] / live, 0.0, w[2] / live])
    return w


def realized(card, panels, cells, origin):
    ttype = str(card["targets"].get("target_type", "level"))
    y = []
    for a, h in cells:
        s = M.full_series(panels, a)
        target = origin + pd.offsets.BDay(h)
        if ttype == "log_return":
            seg = s[(s.index > origin) & (s.index <= target)]
            if seg.empty:
                return None
            y.append(float(seg.sum()))
        else:
            seg = s[s.index <= target]
            if seg.empty or seg.index[-1] <= origin:
                return None
            y.append(float(seg.iloc[-1]))
    return np.array(y)


def monthly(card, panels) -> bool:
    for a in card["targets"]["asset_ids"]:
        s = M.full_series(panels, str(a))
        d = np.diff(s.index.values).astype("timedelta64[D]").astype(float)
        if d.size and np.median(d) > 20:
            return True
    return False


def run(units: pathlib.Path, n_origins: int, gap: int, only: list[str]):
    rows = []
    for u in sorted(units.iterdir()):
        if not (u / "card.toml").exists() or "EXAMPLE" in u.name:
            continue
        card = tomllib.loads((u / "card.toml").read_text(encoding="utf-8"))
        try:
            if monthly(card, u):
                continue
            asof = pd.Timestamp(card["provenance"]["data_cutoff"])
            hmax = max(int(h) for h in card["targets"]["horizons"])
            first = str(sorted(card["targets"]["asset_ids"])[0])
            dates = M.full_series(u, first).index
            dates = dates[dates + pd.offsets.BDay(hmax) <= asof]
            if len(dates) < 80:
                continue
            origins = [dates[-1 - k * gap] for k in range(n_origins) if len(dates) - 1 - k * gap >= 60]
        except M.M0Refused:
            continue
        fam = card.get("task", {}).get("id", u.name).split("-")[1]
        for o in origins:
            c = json.loads(json.dumps(card))
            c["provenance"]["data_cutoff"] = str(o.date())
            try:
                base, cells, _ = M.draws(c, u)
                y = realized(c, u, cells, o)
                if y is None:
                    continue
                cb = components(base, y, c)
                w = weights(c, len(cells))
                for name, v in VARIANTS.items():
                    if only and name not in only and name != "M0":
                        continue
                    s, _, _ = M.draws(c, u, variant=v)
                    cv = components(s, y, c)
                    ratio = np.where(cb > 0, cv / np.where(cb > 0, cb, 1.0), 1.0)
                    score = float(min(4.0, np.dot(w, ratio)))
                    rows.append({"unit": u.name, "family": fam, "origin": str(o.date()),
                                 "variant": name, "score": score})
            except M.M0Refused:
                continue
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("units", type=pathlib.Path)
    ap.add_argument("--origins", type=int, default=10)
    ap.add_argument("--gap", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    a = ap.parse_args()
    df = run(a.units, a.origins, a.gap, a.only)
    if a.csv:
        df.to_csv(a.csv, index=False)
    n_units = df[df.variant == "M0"].unit.nunique()
    print(f"{n_units} cards, {len(df[df.variant == 'M0'])} (card, origin) pairs")
    per_card = df.groupby(["variant", "unit", "family"]).score.mean().reset_index()
    fam = per_card.pivot_table(index="variant", columns="family", values="score", aggfunc="mean")
    summ = pd.DataFrame({
        "mean(pairs)": df.groupby("variant").score.mean(),
        "median": df.groupby("variant").score.median(),
        "mean(cards)": per_card.groupby("variant").score.mean(),
        "cards<1": per_card.assign(b=per_card.score < 1).groupby("variant").b.mean(),
    }).join(fam)
    print(summ.sort_values("mean(cards)").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
