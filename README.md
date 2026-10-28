# Horizona-NX — Reading the News into the Forecast

Probabilistic market forecasting from time series plus a news corpus. Agenthon 2026 (NeurIPS) Track 2 — Forecasting,
Team 210.

A forecasting agent that lets text reshape its uncertainty, not just its numbers. It combines multivariate market time
series (rates, FX, macro, factor returns) with a frozen news corpus to produce full predictive distributions. The numeric
core reproduces the reference baseline (M0) exactly, so any gain is information uplift from text.

- **Forecasting layer** — deterministic scenario engine: trailing-window drift and covariance, seeded per series,
  generating path samples (`agent/forecast_core/m0_exact.py`, `scenario_mixture.py`).
- **Reasoning layer** — a language model reads only text published before the forecast date (as-of filtering) and emits
  bounded volatility multipliers per asset with a written rationale; a deterministic lexicon fallback runs when the model
  is unavailable (`llm_signal.py`, `text_signal.py`, `rationale.py`).
- **Self-check** — schema, monotone quantiles and embargo compliance before writing (`self_check.py`).

## Versions

| Version | Role | Where |
|---|---|---|
| **newscalegate** | Final submission (22 Oct 2026) | `agent/` |
| newscale | best development version (Dev board 1.9190 on 71 cards) | `versions/newscale_to_newscalegate.patch` |

Final image: `ghcr.io/syehsyoh-dokling/qf-forecast@sha256:b7026590da5bee513263e482663e0bd17a1c597a52b7b66e040f0334660aa03b`
(descriptor in `submission/`; the newscale descriptor is kept beside it).

## Layout

```
agent/              Final agent source and Dockerfile
image/              base-image build scripts
versions/           patch from newscale to newscalegate
simulator/          local replica of the leaderboard scorer, grids and calibration evidence
tools/final_check/  scripts used to compare the Final candidates on the practice set
docs/               Final-candidate report
submission/         descriptors and packaging script (the team key is entered at run time, never stored)
data/               the organisers' public Track 2 repository (their licence applies)
```

## Licence

Code: Apache-2.0. `data/` is a copy of the organisers' public material and remains under their licence.
