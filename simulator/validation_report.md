# Validation against live per-card leaderboard scores

Aggregation rule (primary = mean over 71 cards of min(score, 4.0)) applied to the pasted per-card live scores of each upload:

| upload | reported | recomputed | cards |
|---|---|---|---|
| dev-0923 model | 1.2982 | 1.2982 | 71 |
| dev-0924 drift-drop | 1.2624 | 1.2624 | 71 |
| cal-v3 window504 | 1.1773 | 1.1773 | 71 |
| cal-v2 window1260 | 1.2303 | 1.2303 | 71 |
| cal-v1 house-text | 1.2509 | 1.2509 | 71 |
| cal-v4 per-family | 1.1460 | 1.1460 | 71 |
| m0b | 1.1368 | 1.1368 | 71 |
| m0x exact | 1.0000 | 1.0000 | 71 |

## m0b upload (vendored reference CLI walk, F1/F3 scale from the last 504 changes; live 1.1368)

- cards simulated: 71 of 71
- simulated primary **1.1368** vs live **1.1368** (old simulator: 1.1327 on 68 cards + 3 live fills)
- per card |local - live|: exact to the 4 printed decimals (<0.0001) on 70/71, <0.002 on 71, median 0.00002, max 0.0013, correlation 1.000000 (old simulator: 0.998)
- family means live / local: F1 1.4149 / 1.4150, F2 1.0212 / 1.0212, F3 0.9975 / 0.9975, F4 1.0894 / 1.0894
- cards not exact:
  - t2-F1-ai-mom-2024: live 2.0604, local 2.0617 (approximate truth)

## exact M0 upload (m0_exact, no variant; live 1.0000)

- simulated primary **1.0000** vs live 1.0000; per card max |local - live| 0.00004
- live cards not at 1.0000: t2-F3-conundrum-joint-2005 live 0.9985 / local 0.9985, t2-F4-quant-crowding-2007 live 0.9994 / local 0.9994
  (the organizers' M0 uses the card's [targets] list order of cells on these two cards; our agent draws in sorted order)

## Hypotheses tested on m0b (first pass, validate.py; exact = |err| < 0.0002 of 71)

| hypothesis | primary | exact | corr |
|---|---|---|---|
| h-th row, log_return = sum(r) | 1.1325 | 55 | 0.9980 |
| h-th row, log_return = sum(log(1+r))  [adopted] | 1.1364 | 67 | 0.9999 |
| target = as-of + h business days | 1.2526 | 4 | 0.6400 |
| coverage tail instead of pinball | 1.2396 | 0 | 0.8218 |
| composite ratio instead of per-component ratios | 1.1361 | 0 | 0.7717 |

## Truth coverage

- validation cards with truth: 71/71; data-verified (source reproduces the card's own pre-as-of panel exactly): 70; live m0b reproduced (<0.002): 71; **verified (both): 70**
- public-dev cards (not on the board, no live check): 32 cards, truth for 28, data-verified 28
- not verified: t2-F1-ai-mom-2024 (approximate source / no live match); t2-F1-cpi-glidepath-2023 (monthly macro card (as-of ALFRED vintage panel, release-date target): ); t2-F1-sahm-watch-2024 (monthly macro card (as-of ALFRED vintage panel, release-date target): ); t2-F4-covid-nfp-2020 (monthly macro card (as-of ALFRED vintage panel, release-date target): ); t2-F4-cpi-vintage-2022 (monthly macro card (as-of ALFRED vintage panel, release-date target): )
