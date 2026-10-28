#!/usr/bin/env python3
"""Rolling-origin backtest for the Track 2 sampler: CRPS, PIT, coverage.

Sealed outcomes make the official score unmeasurable locally, but every unit
carries its own future: roll the as-of date back inside the panel, forecast,
and compare against the realised rows that follow. This is the standard
rolling-origin evaluation, and it is the honest local proxy for the marginal
half of the composite.

Reported per configuration:
  crps_norm   draws-based CRPS, normalised by sigma_daily*sqrt(h) so units
              and horizons are comparable (lower is better; ~0.55 is what a
              well-calibrated Gaussian random walk would score)
  cover90     fraction of realised values inside [q05, q95] (target 0.90)
  pit_mean    mean probability-integral-transform (target 0.50; higher means
              forecasts sit below realisations - drift missing)
  pit_std     PIT standard deviation (target 0.289 for uniform; smaller means
              OVERdispersed forecast, larger means UNDERdispersed)

    backtest_eval.py [--units N] [--origins K] [--repo DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from forecast_core import (llm_signal, panel_loader, scenario_mixture,  # noqa: E402
                           self_check, text_signal)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


def crps_from_draws(draws: np.ndarray, realised: float) -> float:
    """CRPS estimator from an ensemble: E|X-y| - 0.5 E|X-X'| (O(m log m))."""
    x = np.sort(draws)
    m = len(x)
    term1 = np.abs(x - realised).mean()
    # E|X-X'| via the sorted-sample identity.
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = 2.0 * (weights * x).sum() / (m * m)
    return float(term1 - 0.5 * term2)


try:
    # The organizer's own metric (qfbench2_common, installed from the kit):
    # marginal CRPS + variogram + tail penalty, exactly as the scorer calls it.
    from qfbench2_common.scoring import crps as official_crps
except ImportError:  # pragma: no cover - kit not installed
    official_crps = None

#: Official composite weights and the single-cell renormalisation
#: (track-lead ruling 2026-08-24): on a 1-cell grid the variogram is 0 by
#: construction and the weights become (0.714, 0, 0.286).
WEIGHTS = (0.5, 0.3, 0.2)
TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)


def joint_matrix(draws: pd.DataFrame, assets: list[str],
                 horizons: list[int]) -> np.ndarray:
    """[n_draws, cells] in the scorer's asset-outer / horizon-inner order."""
    n_draws = int(draws["draw"].max()) + 1
    out = np.empty((n_draws, len(assets) * len(horizons)))
    for ai, asset in enumerate(assets):
        for hi, horizon in enumerate(horizons):
            cell = draws[(draws["asset"] == asset) & (draws["horizon"] == horizon)]
            out[:, ai * len(horizons) + hi] = (
                cell.sort_values("draw")["value"].to_numpy()[:n_draws])
    return out


def rw_baseline(history: np.ndarray, horizons: list[int], n_draws: int,
                seed: int = 0) -> np.ndarray:
    """The organizer's text-blind baseline, reproduced from baselines/base.py:
    per asset, drift = mean of first differences, sigma = their std (floored),
    draws i.i.d. across assets AND horizons: last + mu*h + sigma*sqrt(h)*Z.
    Every one of the five official baselines falls through to this offline."""
    rng = np.random.default_rng(seed)
    n_assets = history.shape[1]
    hs = np.asarray(horizons, dtype=float)
    out = np.empty((n_draws, n_assets * len(horizons)))
    for ai in range(n_assets):
        series = history[:, ai]
        diffs = np.diff(series)
        last = float(series[-1])
        mu = float(np.mean(diffs)) if diffs.size else 0.0
        sigma = float(np.std(diffs)) if diffs.size else 0.0
        sigma = max(sigma, abs(last) * 1e-4, 1e-6)
        z = rng.standard_normal((n_draws, len(horizons)))
        out[:, ai * len(horizons):(ai + 1) * len(horizons)] = (
            last + mu * hs[None, :] + sigma * np.sqrt(hs)[None, :] * z)
    return out


def official_components(samples: np.ndarray, y: np.ndarray) -> dict | None:
    if official_crps is None:
        return None
    out = official_crps.crps_composite(samples, y, weights=WEIGHTS,
                                       tail_levels=TAIL_LEVELS,
                                       joint="variogram")
    return {k: float(out[k]) for k in ("marginal", "joint", "tail", "composite")}


def normalised_composite(ours: dict, base: dict, cells: int) -> dict:
    """Each component divided by the baseline's (1.0 = random walk), then
    weighted as the scorer does, with the single-cell renormalisation."""
    weights = {"marginal": 0.5, "joint": 0.3, "tail": 0.2}
    if cells == 1:
        weights = {"marginal": 0.5 / 0.7, "joint": 0.0, "tail": 0.2 / 0.7}
    ratios, composite = {}, 0.0
    for name, w in weights.items():
        if w == 0.0:
            continue
        denom = base[name]
        ratio = ours[name] / denom if denom > 0 else (1.0 if ours[name] == 0 else np.inf)
        ratios[f"{name}_ratio"] = ratio
        composite += w * ratio
    ratios["composite_norm"] = composite
    return ratios


def read_grid(unit: pathlib.Path) -> dict | None:
    card_path = unit / "card.toml"
    if not card_path.is_file():
        return None
    card = tomllib.loads(card_path.read_text(encoding="utf-8", errors="replace"))
    targets = card.get("targets") or card.get("task", {}).get("targets") or {}
    if not targets.get("asset_ids") or not targets.get("horizons"):
        return None
    return {"assets": list(targets["asset_ids"]),
            "horizons": [int(h) for h in targets["horizons"]],
            "target": targets.get("target_type", "level")}


def _signal_for(unit: pathlib.Path, asof: str, assets: list[str],
                horizons: list[int], mode: str) -> dict:
    if mode == "none":
        return {}
    text_dir = unit / "text"
    if mode == "lexicon":
        return text_signal.read_corpus(text_dir, asof)
    return llm_signal.read_corpus(text_dir, asof, assets, horizons, "level")


def evaluate_unit(unit: pathlib.Path, n_origins: int,
                  n_draws: int, vol_adjust: bool = True,
                  text_mode: str = "none",
                  self_correct: bool = False,
                  scale: float = 1.0,
                  drift_shrink: float = 1.0,
                  drift_min_t: float = 0.0,
                  history_window: int = 0,
                  composites: list | None = None) -> list[dict]:
    """Per-cell records (CRPS/PIT/coverage); per-origin official composites
    are appended to `composites` when the organizer kit is importable."""
    if composites is None:
        composites = []
    grid = read_grid(unit)
    if grid is None:
        return []
    full, _ = panel_loader.load(unit / "panels", unit, "2100-01-01")
    if full.empty:
        return []
    if grid["target"] == "log_return":
        # Score what the organizer scores: cumulative returns. The anchor
        # cancels inside CRPS (ours and the baseline both forecast levels of
        # the same cumulative series), so no subtraction is needed here.
        full, _ = panel_loader.cumulative_from_returns(full)
    wide = (full.pivot_table(index="date", columns="asset", values="value")
            .sort_index().ffill())
    assets = [a for a in grid["assets"] if a in wide.columns]
    if not assets:
        return []
    wide = wide[assets].dropna()
    max_h = max(grid["horizons"])
    if len(wide) < scenario_mixture.MIN_HISTORY + max_h + n_origins:
        return []

    # Evenly spaced forecast origins, each leaving max_h realised rows ahead.
    last_origin = len(wide) - max_h - 1
    first_origin = max(scenario_mixture.MIN_HISTORY,
                       last_origin - 252)  # at most a year of origins
    origins = np.linspace(first_origin, last_origin, n_origins).astype(int)

    records = []
    for origin in origins:
        asof_date = wide.index[origin]
        history = full[full["date"] <= asof_date]
        signal = _signal_for(unit, str(asof_date.date()), assets,
                             grid["horizons"], text_mode)
        correction = 1.0
        if self_correct:
            correction, _ = self_check.dispersion_correction(
                history, assets, grid["horizons"])
        draws, _ = scenario_mixture.build(
            history, assets, grid["horizons"], n_draws=n_draws,
            vol_adjust=vol_adjust, signal=signal, extra_widen=correction * scale,
            drift_shrink=drift_shrink, drift_min_t=drift_min_t,
            history_window=history_window)
        sigma_daily = (wide.iloc[max(0, origin - 250):origin + 1]
                       .diff().std().to_numpy())

        # Official composite (marginal + variogram + tail) on the joint
        # sample matrix, and the same for the organizer's random-walk
        # baseline on identical history - the 1.0 line of the leaderboard.
        if official_crps is not None:
            y_cells = np.array([
                float(wide.iloc[min(origin + h, len(wide) - 1)][asset])
                for asset in assets for h in grid["horizons"]])
            ours = official_components(
                joint_matrix(draws, assets, grid["horizons"]), y_cells)
            base = official_components(
                rw_baseline(wide.iloc[:origin + 1].to_numpy(),
                            grid["horizons"], n_draws), y_cells)
            if ours and base:
                composites.append({
                    "unit": unit.name, "cells": len(y_cells), "origin": int(origin),
                    **{f"ours_{k}": v for k, v in ours.items()},
                    **{f"rw_{k}": v for k, v in base.items()},
                    **normalised_composite(ours, base, len(y_cells)),
                })
        for horizon in grid["horizons"]:
            target_row = min(origin + horizon, len(wide) - 1)
            realised = wide.iloc[target_row].to_numpy()
            sub = draws[draws["horizon"] == horizon]
            for ai, asset in enumerate(assets):
                sample = sub[sub["asset"] == asset]["value"].to_numpy()
                scale = float(sigma_daily[ai]) * np.sqrt(horizon)
                if not np.isfinite(scale) or scale <= 0:
                    continue
                y = float(realised[ai])
                records.append({
                    "unit": unit.name, "asset": asset, "horizon": horizon,
                    "crps_norm": crps_from_draws(sample, y) / scale,
                    "pit": float((sample <= y).mean()),
                    "cover90": float(np.quantile(sample, 0.05) <= y
                                     <= np.quantile(sample, 0.95)),
                })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",
                        default="/mnt/c/Users/Saifuddin/Documents/track2-forecasting-public")
    parser.add_argument("--units", type=int, default=6)
    parser.add_argument("--origins", type=int, default=8)
    parser.add_argument("--n-draws", type=int, default=400)
    parser.add_argument("--label", default="current")
    parser.add_argument("--no-fhs", action="store_true",
                        help="disable EWMA vol adjustment (A/B against raw "
                             "block bootstrap)")
    parser.add_argument("--self-correct", action="store_true",
                        help="apply the pre-ship dispersion self-backtest "
                             "per origin (leak-free: uses history <= origin)")
    parser.add_argument("--text-mode", choices=["none", "lexicon", "llm"],
                        default="none",
                        help="uplift protocol: none = statistical floor; "
                             "lexicon = shipped fallback reader; llm = model "
                             "reader via MODEL_ENDPOINT/MODEL_NAME env")
    parser.add_argument("--drift-shrink", type=float, default=1.0,
                        help="scale the historical drift in the forecast centre (1.0 = shipped)")
    parser.add_argument("--drift-min-t", type=float, default=0.0,
                        help="keep a series' drift in full when |t| >= this (0 = off)")
    parser.add_argument("--history-window", type=int, default=0,
                        help="draw the spread from the last N observations only (0 = all)")
    parser.add_argument("--dump", default=None,
                        help="write every origin's composite record to this JSON file")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply forecast dispersion around the drift (1.0 = shipped)")
    args = parser.parse_args()

    units_dir = pathlib.Path(args.repo) / "units"
    units = sorted(p for p in units_dir.iterdir() if p.is_dir())[: args.units]

    all_records = []
    composites: list[dict] = []
    for unit in units:
        records = evaluate_unit(unit, args.origins, args.n_draws,
                                vol_adjust=not args.no_fhs,
                                text_mode=args.text_mode,
                                self_correct=args.self_correct,
                                scale=args.scale,
                                drift_shrink=args.drift_shrink,
                                drift_min_t=args.drift_min_t,
                                history_window=args.history_window,
                                composites=composites)
        all_records += records
        if records:
            frame = pd.DataFrame(records)
            print(f"{unit.name:<38} n={len(frame):>3}  "
                  f"crps={frame['crps_norm'].mean():.4f}  "
                  f"cover90={frame['cover90'].mean():.3f}  "
                  f"pit_std={frame['pit'].std():.3f}", flush=True)
        else:
            print(f"{unit.name:<38} skipped (no grid/short panel)", flush=True)

    if not all_records:
        print("nothing evaluated")
        return 1
    frame = pd.DataFrame(all_records)
    print("\n" + "=" * 68)
    print(f"[{args.label}] {len(frame)} forecasts over {frame['unit'].nunique()} units")
    print(f"  crps_norm  {frame['crps_norm'].mean():.4f}   (lower better)")
    print(f"  cover90    {frame['cover90'].mean():.3f}    (target 0.900)")
    print(f"  pit_mean   {frame['pit'].mean():.3f}    (target 0.500)")
    print(f"  pit_std    {frame['pit'].std():.3f}    (target 0.289; "
          "<: overdispersed, >: underdispersed)")
    if args.text_mode == "llm":
        print(f"  llm reader: {llm_signal.STATS}")
    if args.dump:
        import json as _json
        pathlib.Path(args.dump).write_text(_json.dumps(composites), encoding="utf-8")
    if composites:
        c = pd.DataFrame(composites)
        multi = c[c["cells"] > 1]
        print(f"\n  OFFICIAL METRIC vs the organizer's random-walk baseline "
              f"({len(c)} origins, {c['unit'].nunique()} units; 1.0 = random walk)")
        print(f"  marginal CRPS ratio   {c['marginal_ratio'].mean():.3f}"
              f"   (median {c['marginal_ratio'].median():.3f})")
        if len(multi):
            print(f"  variogram ratio       {multi['joint_ratio'].mean():.3f}"
                  f"   (median {multi['joint_ratio'].median():.3f}; "
                  f"{len(multi)} multi-cell origins)")
        print(f"  tail penalty ratio    {c['tail_ratio'].mean():.3f}"
              f"   (median {c['tail_ratio'].median():.3f})")
        print(f"  COMPOSITE (normalised) {c['composite_norm'].mean():.3f}"
              f"   (median {c['composite_norm'].median():.3f}; "
              f"{(c['composite_norm'] < 1).mean():.0%} of origins beat the walk)")
        per_unit = c.groupby("unit")["composite_norm"].mean().sort_values()
        worst = per_unit.tail(3)
        print("  worst units: " + ", ".join(f"{u} {v:.2f}" for u, v in worst.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
