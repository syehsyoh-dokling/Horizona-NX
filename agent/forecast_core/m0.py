"""The official Track-2 M0 forecaster, vendored verbatim.

Source: Agenthon-2026/track2-forecasting-public, qfbench2_track_forecasting/cli.py (MIT License,
Copyright (c) the Agenthon 2026 organizers). The functions below are copied unchanged; only the
import of ParseLimits is replaced by its two constants (min_draws 200, max_draws 20,000), because
that module pulls in the organizer toolkit, which the image does not ship.

Why verbatim: the leaderboard divides every card by THIS forecaster's error (normalization.py:
"the official M0 baseline ... the baseline is 1.0 by construction"). Same code, same seed (0), same
draw count -> the same forecast -> a score of 1.0 on every card. It is the anchor the next
variants are measured against, one change at a time.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

_MIN_DRAWS = 200        # ParseLimits().min_draws
_MAX_DRAWS = 20_000     # ParseLimits().max_draws

DEFAULT_DRAWS = 500
_RATIONALE_NAME = "forecast_rationale.md"


def _read_panels(panels_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Every parquet under --panels, keyed by filename stem.

    Accepts the contract layout (`/input/panels/*.parquet`) and also tolerates a unit that keeps
    its panels one level up, which is how the shipped exemplar was laid out before this CLI
    existed. Tolerating it here means a card authored either way still runs.
    """
    found = sorted(panels_dir.glob("*.parquet"))
    if not found and panels_dir.parent.is_dir():
        found = sorted(panels_dir.parent.glob("*.parquet"))
    if not found:
        raise SystemExit(f"no .parquet found under {panels_dir} (or its parent)")
    return {p.stem: pd.read_parquet(p) for p in found}


#: Both spellings occur in the shipped cards — the exemplar unit uses `asset_id`, the pilot and
#: prospective batches use `asset`. A reference implementation has to read either, or it works on
#: some cards and not others for a reason that has nothing to do with forecasting.
_ASSET_COLS = ("asset", "asset_id")


def _asset_col(df: pd.DataFrame) -> str | None:
    return next((c for c in _ASSET_COLS if c in df.columns), None)


def _diff_without_gaps(s: pd.Series) -> pd.Series:
    """First differences, with any difference that spans a hole in the data dropped.

    A transfer card ships its target asset as an early window plus a single row at the as-of
    date, with the years between deliberately withheld (the card says so, and says not to
    difference across it). Differenced naively, that hole reads as one day in which the asset
    moved a decade's worth -- on the CNY card it inflated the 5-95% band from under a percent
    to +-7%. The threshold adapts to the panel's own spacing (10x its typical step), so daily
    and monthly panels are both handled and a gapless panel is untouched.
    """
    d = s.diff()
    when = pd.to_datetime(pd.Series(s.index, index=s.index), errors="coerce")
    step = when.diff().dt.days
    if step.notna().sum() == 0:
        return d
    return d.where(step <= max(float(step.median()) * 10.0, 5.0))


def _series(panels: dict[str, pd.DataFrame], asset: str, asof: str) -> pd.Series:
    """The history of one asset up to and including the as-of, from whichever panel holds it."""
    for df in panels.values():
        col = _asset_col(df)
        if col is None:
            continue
        sub = df[df[col].astype(str) == asset]
        if sub.empty:
            continue
        sub = sub.copy()
        # Dates arrive as either strings or datetimes depending on how the panel was written.
        sub["date"] = sub["date"].astype(str).str.slice(0, 10)
        sub = sub[sub["date"] <= asof].sort_values("date")
        if not sub.empty:
            return sub.set_index("date")["value"].astype(float)
    seen = sorted(
        {
            str(v)
            for df in panels.values()
            if (c := _asset_col(df)) is not None
            for v in df[c].unique()
        }
    )
    raise SystemExit(
        f"asset {asset!r} not present in any panel at or before {asof}. "
        f"Panels carry: {', '.join(seen) if seen else '(no asset column found)'}"
    )


def _draw(
    panels: dict[str, pd.DataFrame],
    assets: list[str],
    horizons: list[int],
    asof: str,
    n_draws: int,
    seed: int,
    sd_window: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Joint Gaussian random walk, correlated ACROSS ASSETS via their historical daily changes.

    Drawing each asset independently would score badly on purpose: the composite puts 0.3 on the
    joint variogram term precisely to catch marginals that were stapled together. So the shared
    innovation is drawn from the empirical correlation of daily changes and scaled by sqrt(h).
    """
    rng = np.random.default_rng(seed)
    hist = {a: _series(panels, a, asof) for a in assets}
    diffs = pd.DataFrame({a: _diff_without_gaps(s) for a, s in hist.items()}).dropna()
    if len(diffs) < 30:
        raise SystemExit(f"not enough history to estimate covariance ({len(diffs)} rows)")

    last = np.array([hist[a].iloc[-1] for a in assets], dtype=float)
    # OUR ONLY CHANGE to the vendored M0: sd_window (default None = verbatim M0) takes the scale
    # from the most recent sd_window changes; the correlation still uses the full history.
    sd = (diffs.tail(sd_window) if sd_window else diffs).std().to_numpy(dtype=float)
    corr = diffs.corr().to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    # Nearest-PSD nudge: an empirical correlation can be indefinite after nan_to_num.
    w, v = np.linalg.eigh(corr)
    corr = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    chol = np.linalg.cholesky(corr)

    out = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    for hi, h in enumerate(horizons):
        z = rng.standard_normal((n_draws, len(assets))) @ chol.T
        out[:, :, hi] = last + z * (sd * np.sqrt(h))
    meta = {
        "last": {a: float(last[i]) for i, a in enumerate(assets)},
        "daily_sd": {a: float(sd[i]) for i, a in enumerate(assets)},
        "n_history_rows": int(len(diffs)),
    }
    return out, meta


def _rationale(
    unit_id: str,
    asof: str,
    assets: list[str],
    horizons: list[int],
    n_draws: int,
    stats: dict[str, Any],
    text_dir: pathlib.Path,
) -> str:
    n_docs = len(list(text_dir.glob("*.txt"))) if text_dir.is_dir() else 0
    ladder = "\n".join(
        f"| {a} | {stats['last'][a]:.4f} | {stats['daily_sd'][a]:.4f} | "
        f"{stats['daily_sd'][a] * np.sqrt(h):.4f} | {h} |"
        for a in assets
        for h in horizons
    )
    return f"""# Forecast rationale — {unit_id}

As of **{asof}**, joint distribution over {", ".join(assets)} at horizon(s)
{", ".join(str(h) for h in horizons)} business days. {n_draws} draws.

## Anchor

The last observed value of each series at the as-of, taken from the shipped panels
({stats["n_history_rows"]} rows of overlapping daily history used for the covariance).

## Adjustments

**None.** This is a driftless random walk: the centre is the anchor, unadjusted. Every
adjustment is zero and is listed as such rather than omitted, so the ledger below sums.

## Scale and shape

Per-asset daily standard deviation of first differences, scaled by sqrt(horizon). Gaussian
shape — deliberately not fat-tailed, since nothing here justifies a tail view.

The draws are **joint**: a single innovation vector is drawn per draw from the empirical
correlation of daily changes across assets, so cross-asset structure is preserved rather than
assembled from independent marginals. The composite's variogram term scores exactly that.

## Adjustment ledger

| asset | anchor | daily sd | sd at horizon | horizon (BD) |
|---|---|---|---|---|
{ladder}

Centre = anchor + 0 for every asset and horizon.

## What the text corpus contributed

**Nothing.** {n_docs} document(s) were present at the text path and none was read. This is the
statistical floor a reasoning agent has to beat, not an example of using text — the whole point
of Track 2 is the gap between this and an agent that reads the corpus. A real submission would
use the documents to move the centre, skew the distribution, or widen the tails, and would say
here which document drove which adjustment and by how much.

## What would change this forecast

Any evidence at all. It currently uses none beyond the panel's own volatility.
"""


def run(card: dict[str, Any], panels_dir: pathlib.Path, text_dir: pathlib.Path, asof: str,
        out: pathlib.Path, n_draws: int | None = None, seed: int = 0,
        sd_window: int | None = None) -> None:
    """cli.main() from `card = tomllib.loads(...)` on, unchanged in substance."""
    tgt = card["targets"]
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]
    unit_id = card["task"]["id"]
    card_floor = int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 0) or 0)
    floor = max(card_floor, DEFAULT_DRAWS, _MIN_DRAWS)
    n_draws = max(n_draws or floor, floor)
    if n_draws > _MAX_DRAWS:
        raise SystemExit(f"--n-draws {n_draws} exceeds the contract ceiling {_MAX_DRAWS}")

    panels = _read_panels(panels_dir)
    samples, stats = _draw(panels, assets, horizons, asof, n_draws, seed, sd_window)

    out_dir = out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"draw": d, "asset": asset, "horizon": h, "value": float(samples[d, ai, hi])}
            for d in range(n_draws)
            for ai, asset in enumerate(assets)
            for hi, h in enumerate(horizons)
        ]
    ).to_parquet(out, index=False)

    (out_dir / "forecast_meta.json").write_text(
        json.dumps(
            {
                "unit_id": unit_id,
                "asof": asof,
                "representation": "samples",
                "asset_ids": assets,
                "horizons": horizons,
                "n_draws": n_draws,
                "target": tgt.get("target_type", "level"),
                "rationale": {
                    "file": _RATIONALE_NAME,
                    "method": "joint gaussian random walk, no text",
                },
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / _RATIONALE_NAME).write_text(
        _rationale(unit_id, asof, assets, horizons, n_draws, stats, text_dir)
    )
