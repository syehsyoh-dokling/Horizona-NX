# Track 2 Final - second upload decision on the short-horizon practice set (22 Oct 2026)

## Summary

- The 22 `-ps1` cards were scored with a rebuilt scorer (`t2work/sim/rawscore.py`) that matches the
  official `scoring/scoring.py` components to ~1e-16; every container output below was also re-scored by
  the official CLI (admissible on all 22 cards, identical raw composites).
- **The newscale image has a defect at horizon 2.** `forecast_core/m0_exact.py::panel_steps` implements
  docs/M0-BASELINE.md section 3.7 step 5 with `ratio >= 2`; the specification is **strict** (`ratio > 2`)
  and names this exact case (a 2-day horizon from a Monday counts 1 step, ratio exactly 2.0, declared
  horizon kept). The image collapses every h=2 cell to one panel step (width too narrow by sqrt 2; with an
  h=1 cell present both cells get identical draws). 6 of the 22 practice cards carry an h=2 cell; the Final
  is 1-4 BD from a Monday as-of, so Final cards will too.
- `newscalefix` (one-line fix, configuration unchanged): leaderboard-scale mean **1.4392 -> 1.2156**,
  lower on both halves (odd 1.1187 -> 1.0635, even 1.7598 -> 1.3676), 5 of 6 affected cards better,
  1 worse (+0.41), 16 cards byte-identical; LOO [-0.25, -0.11]; bootstrap P(better) 0.97.
- `newscalegate` (fix + horizon-gated widening, section 11): cells with declared horizon <= 5 BD keep
  M0's own width, longer cells keep the newscale family widenings. Mean **1.2156 -> 1.1105**, lower on
  both halves (odd 1.0635 -> 1.0577, even 1.3676 -> 1.1632), below exact M0 (1.1419) on both halves.
  No Dev unit has a horizon <= 5 BD (minimum is 7), so the Dev result 1.9190 is unchanged for either image.
- **Recommendation: upload `newscalegate` as the second (replacing) Final entry**; `newscalefix` is the
  fallback if the owner prefers the pure bug fix. Both are built, pushed, anonymously pullable (HTTP 200),
  with descriptors and pack scripts under `submissions/2026-10-22/`.

## 1. The practice set

Public repo `Agenthon-2026/track2-forecasting-public` cloned to `t2work/public` (22 Oct). 22 units with
ids ending `-ps1`, split `public-dev`, all `level`, 53 cells:

| group | cards | assets | panel | horizons seen |
|---|---:|---|---|---|
| UST bills | 8 | UST_3M, UST_6M, UST_1Y (+UST_2Y on one F3 card), % p.a. (2 dp) | `rates_daily` | 1, 2, 3, 4 |
| EM FX | 11 | MXN, BRL, ZAR, KRW, INR per USD (ECB crosses) | `em_fx_ecb_daily` (new) | 1, 2, 3, 4 |
| G10 FX | 3 | EUR, JPY, CHF (H.10 quotes) | `g10_fx_daily` | 1, 2, 3, 4 |

Every as-of is a Monday between 2009-08-10 and 2022-12-12; every target is the Tue-Fri of the same week.
Realised values: `practice-answers/<unit>.parquet`. Inventory: `t2work/sim/inventory_ps1.py`.

## 2. Scorer rebuilt (`t2work/sim/rawscore.py`)

- Components exactly as `qfbench2_track_forecasting.scoring._composite`: `crps_marginal`,
  `variogram_score(p=0.5)` (organizers' toolkit `qfbench2-common` 2.6.1 in `t2work/sim/venv`), pinball
  tail at [0.01, 0.05, 0.95, 0.99]; weights (0.5, 0.3, 0.2); single-cell renormalisation (none needed).
- Divisors: section 5 formulas (marginal `mean sd / sqrt(pi)`, tail `0.0648939 x mean sd`, joint
  `sum_{i != j} Var(|D_ij|^(1/2))` via `hyp1f1`) from M0's mean/covariance (trailing 300 steps, the
  **declared horizon as step count** - section 3.7 with the sealed target dates keeps the declared horizon
  on every ps1 cell).
- Verification: trivial forecasts on a 6-cell FX card and a 3-cell rates card scored by
  `scoring/scoring.py score --realized` and by `rawscore.py`: all components equal to ~1e-16. All 22
  container outputs of `newscalefix` and of `newscalegate` re-scored by the official CLI:
  `admissible: true` everywhere, raw composites identical. (Windows: the official CLI refuses files under
  `Documents` as "hard link (st_nlink=2)"; copies in the session scratchpad pass.)
- Caveat: the organizers' `ref_scale.json` are sealed; the scaled numbers use the published recipe.

## 3. Running our image

VPS `/home/webmaster/build/t2ps1`; newscale pulled by its descriptor digest (`sha256:eda3d373...810a`,
412 MB); inputs laid out as the runner mounts them (`/input/card.toml`, `/input/panels/<panel>.parquet`,
`/input/text/`), no answer files in the mount; one container at a time, `--rm --network=none`, no env
overrides (`vps_run_ps1.sh`). 22/22 runs per image (~0.3 s each), engine `m0x`, text off (`vol_mult {}`,
`centre_moves {}`); the House model is never called in this configuration, no NVIDIA key used.
Container `forecast.parquet` equals the local engine's draws to max 2.3e-13 on every card for all three
images (`score_runs.py`). The image's 11 agent files hash equal to the exported repo HEAD
(`t2work/repo/agent`). All images pulled or built here were removed afterwards; the two pre-existing
`qf-forecast:dev-*` images and all `t1*` material were not touched; disk 1.5 GB free as found.

## 4. The finding: horizon 2 collapses to one panel step

`panel_steps` estimates target = as-of + h BD, spacing = mean calendar gap (~1.4 d on a business-daily
panel), `n = round(days / spacing)`, and overrides the declared horizon when `ratio >= 2`. For h=2 from a
Monday: 2/1.4 -> n = 1, ratio 2.0 -> override to 1 step. Section 3.7 step 5: "The declared horizon wins
unless `ratio > 2`. The comparison is strict because on a business-day panel ... a 2-day horizon from a
Monday, Tuesday or Wednesday counts 1 step, and a 1-day horizon from a Friday counts 2. Both are short
horizons stated correctly ... so the declared horizon is kept." Container logs of the as-is image confirm
`UST_1Y@2 -> 1`, `UST_3M@2 -> 1`, `ZAR@2 -> 1`, `EUR@2 -> 1`, `MXN/BRL/ZAR@2 -> 1`, `UST_6M@2 -> 1`.
On the two {1, 2} cards the as-is image emitted identical draws for both cells: scaled 3.92 -> 1.32 and
3.16 -> 1.61 after the fix. `check_dev_h2.py`: none of the 103 original units (71 validation + 32
public-dev) has a horizon <= 5 BD, so the fix cannot change the Dev board.

## 5. Results: newscale (as is) vs newscalefix vs M0 vs plain RW

### Per-card scores (22 short-horizon practice cards)

Raw = the official scorer's composite in the card's own units (`raw_unrankable`); scaled = raw components divided by M0's expected error (docs/M0-BASELINE.md section 5, declared-horizon steps), clip 8 — the leaderboard's quantity. `newscale` = the Final entry as its image runs (VPS container output); `newscalefix` = the same image with the one-line section-3.7 fix (VPS container output); `M0` = the organizers' text-blind reference row (exact M0, fixed engine); `RW` = fix + plain random walk (no family widening) for comparison.

| card | fam | panel | horizons | raw newscale | raw newscalefix | raw M0 | scaled newscale | scaled newscalefix | scaled M0 | scaled RW |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `t2-F1-curve-ust1y-001-ps1` **(h=2)** | F1 | rates | [2, 3] | 0.014687 | 0.011765 | 0.012457 | 2.5306 | 1.9233 | 2.0859 | 2.0899 |
| `t2-F1-curve-ust3m-002-ps1` **(h=2)** | F1 | rates | [2, 3] | 0.010107 | 0.008291 | 0.009086 | 2.4466 | 1.8817 | 2.0675 | 2.0670 |
| `t2-F1-fx-eur-020-ps1` | F1 | g10_fx | [1, 4] | 0.004955 | 0.004955 | 0.005081 | 0.6139 | 0.6139 | 0.6164 | 0.5968 |
| `t2-F1-fx-inr-010-ps1` | F1 | em_fx | [1, 4] | 0.129440 | 0.129440 | 0.105747 | 1.0503 | 1.0503 | 0.8634 | 0.9365 |
| `t2-F1-fx-krw-011-ps1` | F1 | em_fx | [1, 4] | 2.516361 | 2.516361 | 2.960326 | 0.5918 | 0.5918 | 0.6871 | 0.6570 |
| `t2-F1-fx-zar-009-ps1` **(h=2)** | F1 | em_fx | [1, 2] | 0.109017 | 0.059261 | 0.059650 | 3.9241 | 1.3199 | 1.3268 | 1.2287 |
| `t2-F2-curve-ust1y-003-ps1` | F2 | rates | [3, 4] | 0.005976 | 0.005976 | 0.003952 | 0.6389 | 0.6389 | 0.3915 | 0.4001 |
| `t2-F2-curve-ust6m-004-ps1` | F2 | rates | [1, 4] | 0.040935 | 0.040935 | 0.054645 | 1.4970 | 1.4970 | 1.8420 | 1.1601 |
| `t2-F2-fx-brl-012-ps1` | F2 | em_fx | [3, 4] | 0.017899 | 0.017899 | 0.017753 | 0.8970 | 0.8970 | 0.9710 | 1.0322 |
| `t2-F2-fx-eur-021-ps1` **(h=2)** | F2 | g10_fx | [2, 4] | 0.008504 | 0.009674 | 0.011491 | 1.9752 | 2.3860 | 3.1146 | 3.2721 |
| `t2-F2-fx-krw-015-ps1` | F2 | em_fx | [1, 3] | 3.323743 | 3.323743 | 2.249037 | 1.2700 | 1.2700 | 0.7633 | 0.7734 |
| `t2-F2-fx-mxn-013-ps1` | F2 | em_fx | [1, 4] | 0.081081 | 0.081081 | 0.046242 | 1.1539 | 1.1539 | 0.6284 | 0.6315 |
| `t2-F2-fx-zar-014-ps1` | F2 | em_fx | [1, 4] | 0.080084 | 0.080084 | 0.092601 | 1.0580 | 1.0580 | 1.2259 | 1.2488 |
| `t2-F3-curve-ust3m-ust6m-ust1y-006-ps1` | F3 | rates | [3] | 0.006849 | 0.006849 | 0.006565 | 1.0611 | 1.0611 | 1.0176 | 1.0749 |
| `t2-F3-curve-ust3m-ust6m-ust1y-ust2y-005-ps1` | F3 | rates | [3] | 0.019779 | 0.019779 | 0.019470 | 1.0583 | 1.0583 | 1.0147 | 1.0139 |
| `t2-F3-fx-eur-jpy-chf-022-ps1` | F3 | g10_fx | [3] | 0.114851 | 0.114851 | 0.117425 | 0.7802 | 0.7802 | 0.7962 | 0.7567 |
| `t2-F3-fx-krw-inr-zar-017-ps1` | F3 | em_fx | [4] | 0.570730 | 0.570730 | 0.537584 | 0.5101 | 0.5101 | 0.4668 | 0.4591 |
| `t2-F3-fx-mxn-brl-zar-016-ps1` **(h=2)** | F3 | em_fx | [2, 4] | 0.049453 | 0.044846 | 0.043876 | 0.5530 | 0.5527 | 0.5091 | 0.4900 |
| `t2-F4-curve-ust3m-008-ps1` | F4 | rates | [1, 3] | 0.016105 | 0.016105 | 0.009603 | 1.6125 | 1.6125 | 0.7607 | 0.6963 |
| `t2-F4-curve-ust6m-007-ps1` **(h=2)** | F4 | rates | [1, 2] | 0.009728 | 0.006079 | 0.002618 | 3.1620 | 1.6072 | 0.5697 | 0.5720 |
| `t2-F4-fx-brl-019-ps1` | F4 | em_fx | [3, 4] | 0.053479 | 0.053479 | 0.072485 | 1.5246 | 1.5246 | 2.8330 | 2.7319 |
| `t2-F4-fx-mxn-018-ps1` | F4 | em_fx | [1, 4] | 0.106087 | 0.106087 | 0.040488 | 1.7537 | 1.7537 | 0.5708 | 0.5783 |

### Means (scaled, clip 8)

| group | n | newscale (as is) | newscalefix | M0 reference | RW (fix, no widening) | RW t8 | newscale x0.7 (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **all 22** | 22 | 1.4392 | 1.2156 | 1.1419 | 1.1121 | 1.1214 | 1.0888 |
| odd cards (1,3,5,...) | 11 | 1.1187 | 1.0635 | 1.0742 | 1.0636 | 1.0799 | 0.9118 |
| even cards (2,4,6,...) | 11 | 1.7598 | 1.3676 | 1.2096 | 1.1607 | 1.1628 | 1.2658 |
| family F1 | 6 | 1.8595 | 1.2302 | 1.2745 | 1.2626 | 1.2417 | 1.1956 |
| family F2 | 7 | 1.2129 | 1.2716 | 1.2767 | 1.2169 | 1.2253 | 1.2070 |
| family F3 | 5 | 0.7925 | 0.7925 | 0.7609 | 0.7589 | 0.7590 | 0.7492 |
| family F4 | 4 | 2.0132 | 1.6245 | 1.1836 | 1.1446 | 1.2120 | 1.1461 |
| UST bills (rates_daily) | 8 | 1.7509 | 1.4100 | 1.2187 | 1.1343 | 1.1228 | 1.0540 |
| EM FX (em_fx_ecb_daily) | 11 | 1.2988 | 1.0620 | 0.9860 | 0.9788 | 1.0027 | 0.9902 |
| G10 FX (g10_fx_daily) | 3 | 1.1231 | 1.2601 | 1.5091 | 1.5419 | 1.5524 | 1.5429 |
| cards with an h=2 cell | 6 | 2.4319 | 1.6118 | 1.6122 | 1.6199 | 1.5978 | 1.5363 |
| cards without h=2 | 16 | 1.0670 | 1.0670 | 0.9656 | 0.9217 | 0.9427 | 0.9210 |

Raw-composite means per family are in the cards' own units (KRW/USD ~1000 vs EUR/USD ~1.1 vs bills in % p.a.) and are not comparable across cards; they are listed only because the brief asks for them:

| family | raw newscale | raw newscalefix | raw M0 |
|---|---:|---:|---:|
| F1 (n=6) | 0.46409 | 0.45501 | 0.52539 |
| F2 (n=7) | 0.50832 | 0.50848 | 0.35367 |
| F3 (n=5) | 0.15233 | 0.15141 | 0.14498 |
| F4 (n=4) | 0.04635 | 0.04544 | 0.03130 |

### Candidate vs the current entry (scaled, clip 8)

| candidate | mean diff | odd half | even half | cards better / worse | leave-one-out range of mean diff | bootstrap P(better) | largest single-card gain / loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| **vs newscale (as is)** | | | | | | | |
| m0_fixed | -0.2973 | -0.0445 | -0.5501 | 14 / 8 | [-0.3738, -0.1878] | 0.95 | -2.597 / +1.308 |
| newscale_fixed | -0.2237 | -0.0552 | -0.3921 | 5 / 1 | [-0.2539, -0.1103] | 0.97 | -2.604 / +0.411 |
| rw_fixed | -0.3271 | -0.0551 | -0.5990 | 16 / 6 | [-0.4044, -0.2143] | 0.96 | -2.695 / +1.297 |
| rw_t8_fixed | -0.3179 | -0.0388 | -0.5969 | 16 / 6 | [-0.3998, -0.2054] | 0.95 | -2.680 / +1.403 |
| rw_t5_fixed | -0.3027 | -0.0308 | -0.5746 | 14 / 5 | [-0.3912, -0.1931] | 0.94 | -2.604 / +1.555 |
| ns_x0.7_fixed | -0.3504 | -0.2069 | -0.4939 | 15 / 7 | [-0.4229, -0.2612] | 0.99 | -2.225 / +1.171 |
| **vs newscalefix** | | | | | | | |
| newscale_asis | +0.2237 | +0.0552 | +0.3921 | 1 / 5 | [+0.1103, +0.2539] | 0.02 | -0.411 / +2.604 |
| m0_fixed | -0.0736 | +0.0107 | -0.1580 | 11 / 11 | [-0.1394, -0.0208] | 0.75 | -1.183 / +1.308 |
| rw_fixed | -0.1034 | +0.0001 | -0.2069 | 14 / 8 | [-0.1658, -0.0524] | 0.83 | -1.175 / +1.207 |
| rw_t8_fixed | -0.0942 | +0.0164 | -0.2048 | 14 / 8 | [-0.1655, -0.0435] | 0.79 | -1.158 / +1.403 |
| rw_t5_fixed | -0.0790 | +0.0244 | -0.1825 | 11 / 5 | [-0.1538, -0.0279] | 0.75 | -1.153 / +1.490 |
| ns_x0.7_fixed | -0.1268 | -0.1517 | -0.1018 | 14 / 8 | [-0.1690, -0.0970] | 0.94 | -0.751 / +0.761 |

### Width scan (fixed engine, zero drift, normal innovations; mean scaled score, clip 8)

| group | x0.5 | x0.6 | x0.7 | x0.8 | x0.9 | x1.0 | x1.1 | x1.25 | x1.5 | x2.0 | x3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all 22 (n=22) | 1.451 | 1.320 | 1.225 | 1.166 | 1.130 | **1.112** | 1.115 | 1.144 | 1.240 | 1.518 | 2.283 |
| F1 (n=6) | 1.303 | 1.234 | 1.194 | **1.194** | 1.222 | 1.263 | 1.323 | 1.435 | 1.651 | 2.157 | 3.356 |
| F2 (n=7) | 1.768 | 1.582 | 1.438 | 1.341 | 1.265 | 1.217 | **1.200** | 1.202 | 1.272 | 1.521 | 2.288 |
| F3 (n=5) | 0.805 | 0.769 | 0.750 | **0.739** | 0.746 | 0.759 | 0.775 | 0.813 | 0.894 | 1.085 | 1.526 |
| F4 (n=4) | 1.926 | 1.680 | 1.495 | 1.354 | 1.238 | 1.145 | 1.082 | 1.019 | **0.999** | 1.093 | 1.613 |
| UST bills (n=8) | 1.094 | 1.047 | **1.033** | 1.048 | 1.087 | 1.134 | 1.188 | 1.290 | 1.487 | 1.939 | 2.974 |
| EM FX (n=11) | 1.480 | 1.306 | 1.173 | 1.085 | 1.021 | 0.979 | **0.967** | 0.975 | 1.040 | 1.269 | 1.978 |
| G10 FX (n=3) | 2.298 | 2.101 | 1.931 | 1.780 | 1.646 | 1.542 | 1.465 | 1.373 | 1.313 | **1.304** | 1.559 |

### Width calibration of M0's own distribution at 1-4 BD (z = realised minus M0 mean, in M0 sd units)

| group | cells | mean abs z (normal: 0.798) | rms z (1.0) | inside 50 % band (0.50) | inside 90 % band (0.90) |
|---|---:|---:|---:|---:|---:|
| all | 53 | 0.753 | 0.938 | 0.51 | 0.89 |
| F1 | 12 | 0.636 | 0.828 | 0.67 | 0.92 |
| F2 | 14 | 0.902 | 1.118 | 0.50 | 0.79 |
| F3 | 19 | 0.661 | 0.778 | 0.47 | 0.95 |
| F4 | 8 | 0.885 | 1.081 | 0.38 | 0.88 |
| em_fx | 27 | 0.770 | 0.930 | 0.52 | 0.93 |
| g10_fx | 7 | 0.781 | 0.928 | 0.29 | 0.86 |
| rates | 19 | 0.718 | 0.951 | 0.58 | 0.84 |

## 6. Diagnosis per family (width and centre at 1-4 BD)

- **Width.** M0's textbook width - trailing-300 daily step sd x sqrt(h), one joint path - is calibrated at
  these horizons (rms z 0.94; 51 % / 89 % coverage of the 50 % / 90 % bands; tables above). The pooled
  optimum of a uniform multiplier on M0 is x1.0 (1.112; x1.1 1.115). The family widenings in newscale
  (F2 x1.5, F3 x1.25, F4 x3.0) were tuned on long-horizon Dev cards chosen around shocks; at 1-4 BD they
  cost: F4 x3.0 scores 1.613 against 0.999 at x1.5 and 1.145 at x1.0 (4 cards). Per-family optima (F1 x0.8,
  F2 x1.1, F3 x0.8, F4 x1.5; bills x0.7, EM FX x1.1, G10 x2.0) disagree in direction across 3-8-card groups
  and are read as noise, not as a rule to ship. The principled consequence - keep M0's width at short
  horizons, keep the measured widenings at long ones - is the horizon gate of section 11.
- **Centre.** M0's 300-day mean drift over 1-4 steps is below 0.05 sd on every card; `drift=zero` vs M0
  drift moves the pooled mean by <= 0.03. The last observation is the right short-horizon centre; no
  carry / momentum term was tested (the brief's established facts rule out centre-moving overlays).
- **Tails.** Student-t(8)/t(5) innovations at fixed covariance are within +-0.02 of normal everywhere.
- Plain RW at all horizons (drop t5 and zero-drift too) is flat on the odd half vs newscalefix (+0.0001)
  and fails the brief's criterion; the gate (which keeps t5 / zero drift) does not.

## 7. Recall caveat

The 22 cards resolve between 2009 and 2022, inside the House model's knowledge window (cutoff mid-2025),
so any model-involved score would have to be read as possibly recalled. **No score in this report involves
the model:** the configuration is text-off (`T2_M0_TEXTVOL=0`, `T2_M0_TEXTMODE=""`,
`T2_LLM_FALLBACK=none`), containers ran with `--network=none`, every run log shows `vol_mult {}` /
`centre_moves {}`, and the NVIDIA key was not used. Every number is a deterministic function of the card's
own panel, as it will be on the Final. Scaled numbers rely on the reconstructed divisor (section 2).

## 8. Audit

- `newscalefix`: `diff -r t2work/repo/agent t2work/sim/agent_fixed` = one line
  (`forecast_core/m0_exact.py:113`, `ratio >= 2` -> `ratio > 2`). Inside the built image the file's
  sha256 (`a1061cb4...bf13b`) equals the local fixed file; environment identical to newscale.
- `newscalegate`: `diff -r t2work/repo/agent t2work/sim/agent_gate` = the same line plus one block in
  `m0_exact.draws` (per-cell `widen_gate` rule, 11 lines, horizon-only logic); `agent.py` unchanged (the
  gate enters through the existing `T2_M0_VARIANT="widen_gate=5"` key=value parsing). Image file sha256
  `aeae14d1...7021` equals the local file.
- `grep` over both shipped trees: no unit-id literal (`t2-F?-...`), no `ps1`, no `practice`, no
  outcome lookup. "realised" appears only in `self_check.py` (in-sample check on pre-as-of panel rows,
  disabled, not on the `m0x` path) and in a rationale string. Analysis scripts under `t2work/sim/` read the
  answer files; none is in any image.
- No Team Key handled, no upload, no CodaBench action. The pack scripts only forward `~/.team_key` to
  `qfbench2 submission pack`, as the 13 Oct script does.

## 9. Recommendation

Upload `newscalegate` (section 11) as the second Final entry (replaces newscale) before 26 Oct 11:59 UTC.
Fallback: `newscalefix`. Expected Final effect of either: corrected width and joint structure on every h=2
cell; the gate additionally forecasts every 1-4 BD cell with M0's calibrated width while keeping the Dev-
measured family rules on any longer cell. Expected Dev effect: none (no Dev unit has a horizon <= 5 BD).

## 10. Artefacts

| item | newscalefix | newscalegate |
|---|---|---|
| image tag | `ghcr.io/syehsyoh-dokling/qf-forecast:final-newscalefix-20261022` | `ghcr.io/syehsyoh-dokling/qf-forecast:final-newscalegate-20261022` |
| digest | `sha256:d5429689363d9746c8c1a0e8957ce5a3633767d6d1a0dfc693994499173a3650` | `sha256:b7026590da5bee513263e482663e0bd17a1c597a52b7b66e040f0334660aa03b` |
| anonymous pull | token 200; digest manifest 200 (sha256 of body = digest); tag manifest 200 | same, verified 22 Oct |
| recipe | `t2work/sim/Dockerfile.newscalefix` + `t2work/sim/agent_fixed/forecast_core/m0_exact.py`; `vps_build_fix.sh` | `t2work/sim/Dockerfile.newscalegate` + `t2work/sim/agent_gate/forecast_core/m0_exact.py`; `vps_build_gate.sh` |
| descriptor | `submissions/2026-10-22/submission_t2_final_newscalefix.json` | `submissions/2026-10-22/submission_t2_final_newscalegate.json` |
| pack script | `submissions/2026-10-22/submit_track2_final_newscalefix.ps1` | `submissions/2026-10-22/submit_track2_final_newscalegate.ps1` |
| container outputs | `t2work/sim/out_newscalefix/`, `runs_newscalefix.json` | `t2work/sim/out_newscalegate/`, `runs_newscalegate.json` |

Base of both: `ghcr.io/syehsyoh-dokling/qf-forecast@sha256:eda3d373...810a` (newscale) + `COPY m0_exact.py`
(+ `ENV T2_M0_VARIANT="widen_gate=5"` for the gate). As-is run: `t2work/sim/out_newscale/`,
`runs_newscale.json`. Analysis: `compare2_ps1.py` -> `compare2_ps1.json`, `compare_gate_ps1.py` ->
`compare_gate_ps1.json`, `report_tables.md`, `check_dev_h2.py`, `check_dev_gate.py`, `debug_clip.py`
(how the defect was found). Superseded first pass (`grid_ps1.py`, `compare_ps1.py`) built the divisor with
the buggy step count; their outputs were deleted. Follow-up for the source repo (owner's call): apply the
`m0_exact.py` changes to `syehsyoh-dokling/Time-Series-Forecasting`, add both Dockerfiles under `image/`
and both descriptors under `submission/`.

## 11. Horizon-gated widening (`newscalegate`) - requested by the owner, 22 Oct

**Rule** (generic, horizon only, in `m0_exact.draws`): with `widen_gate=G`, a cell whose declared horizon
is <= G business days has the family `widen` factor undone (M0's own width); cells beyond G keep it. t5
innovations and zero drift of the newscale family rules are kept at all horizons. Textbook rationale: the
calibration in section 6 (M0 width at 1-4 BD: rms z 0.94, 89 % inside the 90 % band) says the short-horizon
width needs no widening; the Dev measurements say the widenings beat M0 at the long horizons they were
measured on. Set in the image through the existing `T2_M0_VARIANT` parsing (`widen_gate=5`).

**Sensitivity of the gate.** On the practice set every horizon is <= 4 BD, so G = 5, 10 and 20 give
byte-identical draws (max |diff| 0.0) and identical scores; they differ only on Dev-like cards:
`check_dev_gate.py` - minimum horizon over the 103 original units is 7 BD; G = 5 touches **0** units,
G = 10 and G = 20 each touch **1** validation unit (`t2-F4-factor-joint-credit-2007`, horizon [7]).
G = 5 is therefore the only gate that leaves the Dev result exactly unchanged; it is the one built.

**Per-card scaled scores (clip 8), fixed engine; `gate5` = container output of the built image**

| card | fam | panel | hor | newscalefix | gate5 (=10=20) | M0 | plain RW |
|---|---|---|---|---:|---:|---:|---:|
| t2-F1-curve-ust1y-001-ps1 | F1 | rates | [2,3] | 1.9233 | 1.9233 | 2.0859 | 2.0899 |
| t2-F1-curve-ust3m-002-ps1 | F1 | rates | [2,3] | 1.8817 | 1.8817 | 2.0675 | 2.0670 |
| t2-F1-fx-eur-020-ps1 | F1 | g10_fx | [1,4] | 0.6139 | 0.6139 | 0.6164 | 0.5968 |
| t2-F1-fx-inr-010-ps1 | F1 | em_fx | [1,4] | 1.0503 | 1.0503 | 0.8634 | 0.9365 |
| t2-F1-fx-krw-011-ps1 | F1 | em_fx | [1,4] | 0.5918 | 0.5918 | 0.6871 | 0.6570 |
| t2-F1-fx-zar-009-ps1 | F1 | em_fx | [1,2] | 1.3199 | 1.3199 | 1.3268 | 1.2287 |
| t2-F2-curve-ust1y-003-ps1 | F2 | rates | [3,4] | 0.6389 | 0.4001 | 0.3915 | 0.4001 |
| t2-F2-curve-ust6m-004-ps1 | F2 | rates | [1,4] | 1.4970 | 1.1601 | 1.8420 | 1.1601 |
| t2-F2-fx-brl-012-ps1 | F2 | em_fx | [3,4] | 0.8970 | 1.0322 | 0.9710 | 1.0322 |
| t2-F2-fx-eur-021-ps1 | F2 | g10_fx | [2,4] | 2.3860 | 3.2721 | 3.1146 | 3.2721 |
| t2-F2-fx-krw-015-ps1 | F2 | em_fx | [1,3] | 1.2700 | 0.7734 | 0.7633 | 0.7734 |
| t2-F2-fx-mxn-013-ps1 | F2 | em_fx | [1,4] | 1.1539 | 0.6315 | 0.6284 | 0.6315 |
| t2-F2-fx-zar-014-ps1 | F2 | em_fx | [1,4] | 1.0580 | 1.2488 | 1.2259 | 1.2488 |
| t2-F3-curve-ust3m-ust6m-ust1y-006-ps1 | F3 | rates | [3] | 1.0611 | 1.1044 | 1.0176 | 1.0749 |
| t2-F3-curve-ust3m-ust6m-ust1y-ust2y-005-ps1 | F3 | rates | [3] | 1.0583 | 1.0333 | 1.0147 | 1.0139 |
| t2-F3-fx-eur-jpy-chf-022-ps1 | F3 | g10_fx | [3] | 0.7802 | 0.7581 | 0.7962 | 0.7567 |
| t2-F3-fx-krw-inr-zar-017-ps1 | F3 | em_fx | [4] | 0.5101 | 0.4246 | 0.4668 | 0.4591 |
| t2-F3-fx-mxn-brl-zar-016-ps1 | F3 | em_fx | [2,4] | 0.5527 | 0.4763 | 0.5091 | 0.4900 |
| t2-F4-curve-ust3m-008-ps1 | F4 | rates | [1,3] | 1.6125 | 0.7607 | 0.7607 | 0.6963 |
| t2-F4-curve-ust6m-007-ps1 | F4 | rates | [1,2] | 1.6072 | 0.5697 | 0.5697 | 0.5720 |
| t2-F4-fx-brl-019-ps1 | F4 | em_fx | [3,4] | 1.5246 | 2.8330 | 2.8330 | 2.7319 |
| t2-F4-fx-mxn-018-ps1 | F4 | em_fx | [1,4] | 1.7537 | 0.5708 | 0.5708 | 0.5783 |

**Means (scaled, clip 8)**

| group | n | newscalefix | gate5 (=10=20) | M0 | plain RW |
|---|---:|---:|---:|---:|---:|
| **all 22** | 22 | 1.2156 | **1.1105** | 1.1419 | 1.1121 |
| odd cards | 11 | 1.0635 | 1.0577 | 1.0742 | 1.0636 |
| even cards | 11 | 1.3676 | 1.1632 | 1.2096 | 1.1607 |
| F1 | 6 | 1.2302 | 1.2302 | 1.2745 | 1.2626 |
| F2 | 7 | 1.2716 | 1.2169 | 1.2767 | 1.2169 |
| F3 | 5 | 0.7925 | 0.7593 | 0.7609 | 0.7589 |
| F4 | 4 | 1.6245 | 1.1836 | 1.1836 | 1.1446 |
| UST bills | 8 | 1.4100 | 1.1042 | 1.2187 | 1.1343 |
| EM FX | 11 | 1.0620 | 0.9957 | 0.9860 | 0.9788 |
| G10 FX | 3 | 1.2601 | 1.5480 | 1.5091 | 1.5419 |

**gate5 vs newscalefix:** mean -0.1051 (odd -0.0057, even -0.2045); 11 cards better / 5 worse (6 F1
cards identical: F1 has no `widen`); LOO range [-0.1724, -0.0538]; bootstrap P(better) 0.83; largest
single-card gain -1.183 (ust3m-008), largest loss +1.308 (brl-019).
**gate5 vs exact M0:** mean -0.0315 (odd -0.0165, even -0.0464); 9 better / 9 worse (4 F4 cards identical:
with the gate F4 is exactly M0 at <= 5 BD); LOO [-0.0419, -0.0005] - all 22 leave-one-out means below M0;
P(better) 0.82; largest gain -0.682, largest loss +0.187. The gate is the only candidate below M0 on both
halves (plain RW: LOO max +0.0013).

**Checks.** Gate engine with the gate off reproduces the fixed engine exactly (max |diff| 0.0). Built image
(`final-newscalegate-20261022`) run on the VPS, 22/22, log shows `variant {"widen_gate": 5.0, "widen": ...}`
with `vol_mult {}`, `centre_moves {}`; container draws = local gate engine to 2.3e-13; official CLI
admissible on all 22 with identical raw composites (e.g. ust3m-008 0.016105 -> 0.009603,
mxn-018 0.106087 -> 0.040488, krw-015 3.3237 -> 2.2913). Dev unchanged (0 units with horizon <= 5 BD).
