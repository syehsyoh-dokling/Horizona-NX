# T2 simulator rebuild - progress (ALL MILESTONES DONE, 1 Oct)

Re-run everything (PowerShell tool; Git Bash mangles /home paths):
  wsl -e bash /mnt/c/Users/Saifuddin/AppData/Local/Temp/claude/C--Users-Saifuddin-Documents-Agenthon-2026/1c8bb465-408f-4455-b5b2-84ff918c0701/scratchpad/t2sim/run_all.sh
Single variant:
  .../python .../t2sim/t2_sim.py score --variant t_nu=8,drift=0.5 [--set all] [--cards]
  .../python .../t2sim/t2_sim.py score --engine m0b --set all

## Files
- repo/                  origin/main of track2-forecasting-public (git archive; nothing modified in the checkout)
- recovered/             old simulator scripts + pasted per-card LIVE scores from the session transcript (scan_log.py)
- live_scores.json       per-card live scores of 8 uploads (extract_live.py)
- public_series/         FRED (H.15, H.10) + Ken French downloads, no API key (fetch_public.py)
- t2_core.py             scorer + engines (m0x variants, m0b) + organizers' M0 cell order
- build_truth.py         -> truth.json
- t2_sim.py              score / validate / table  -> validation_report.md, variant_table.md, results.json
- validate.py, probe_*.py  first-pass hypothesis tests (kept for the record; validation.json)

## Results
- m0b simulated 1.1368 vs live 1.1368; 70/71 cards exact to 4 dp, max error 0.0013 (ai-mom-2024, approximate truth).
- exact M0 simulated 1.0000; the two live exceptions (0.9985, 0.9994) reproduced exactly.
- verified truth: 70/71 validation cards (71/71 have truth). Public-dev: 28/32 (4 monthly cards not rebuilt).
- variant table: see variant_table.md.

## Rules pinned by the live per-card scores
- per-component ratio, variogram p=0.5, pinball tail, single-cell reweighting, clip 4.0
- target = h-th panel row after the as-of; log_return target = sum(log(1+r)) of the next h rows
- organizers' M0 uses card-list cell order on conundrum-joint-2005 and quant-crowding-2007
- divergence-2014 (rates+FX calendars differ): common target dates 2015-01-02 / 2015-04-06 (pinned by live score)
