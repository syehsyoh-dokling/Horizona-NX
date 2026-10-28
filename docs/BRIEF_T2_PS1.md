# Brief: Track 2 second Final upload - decide on the short-horizon practice set (owner, 22 Oct 2026)

Organisers (18 Oct): a short-horizon PRACTICE SET is published - 22 historical cards at horizons of 1-4 business
days "shaped like the cards that rank the Final", ids ending `-ps1`, realised values in `practice-answers/`:
8 on US T-bills (3m, 6m, 1y), 11 on MXN/BRL/ZAR/KRW/INR (new panel `em_fx_ecb_daily`), 3 on G10 FX. README
section "short-horizon practice set" shows how to score a card locally (raw scores, not the board scale):
https://github.com/Agenthon-2026/track2-forecasting-public#short-horizon-practice-set
Each team may make a SECOND Final upload; the later one replaces the entry. Close: 26 Oct 11:59 UTC (18:59 WIB).

Our entry: `newscale` (descriptor `submissions/2026-10-13/submission_t2_final_newscale.json`, pack script
`submit_track2_final_newscale.ps1`). Source + simulator: GitHub `syehsyoh-dokling/Time-Series-Forecasting`
(local clone `C:/Users/Saifuddin/Documents/Time-Series-Forecasting` holds only .git by the owner's wish - do NOT
check files out there; export with `git -C <dir> archive HEAD | tar -x -C C:/Users/Saifuddin/Documents/Agenthon-2026/t2work/repo`).
Established facts (do not re-derive): on the 71 Dev cards the no-outcome-knowledge floor is ~1.9 on the new scale;
recall of outcomes is void on the Final window and is NOT a method (organisers screen for it); do not spend time on
sub-1.0 "gambles". The practice set is different: 1-4 day horizons, T-bills and EM FX, resolved in the past but
possibly inside the House model's knowledge - so any local score must be reported with the recall caveat
(check the card dates against the model cutoff, mid-2025).

## Work
1. Clone the public repo (`Agenthon-2026/track2-forecasting-public`) into `t2work/public`; read the README practice
   section and the 22 `-ps1` cards; note horizons, panels, as-of dates, realised values.
2. Rebuild the local scorer per the README (raw score) in `t2work/sim`; verify it on a trivial forecast.
3. Run OUR newscale image on the 22 cards (pull `ghcr.io/syehsyoh-dokling/<repo>@<digest>` from the descriptor on
   the VPS `ssh biznet-webmaster`, dir `/home/webmaster/build/t2ps1`; disk is tight (~1.8 GB free): check `df -h`,
   use `--rm`, remove your own images after; at most ONE run at a time; NVIDIA key `/home/webmaster/.agenthon_nvidia_key`,
   endpoint https://integrate.api.nvidia.com/v1, model nvidia/nemotron-3-super-120b-a12b; T1 lanes are running
   there - do not touch t1_* files, t1all/, or t1all containers). Score per card and the mean; also the M0 /
   naive baseline the README describes, for scale.
4. Diagnose per family (T-bills, EM FX, G10 FX): is the width calibrated at 1-4 day horizons (too wide / too
   narrow)? is the centre a sensible short-horizon forecast (random walk / carry / last value)? The agent was tuned
   on long-horizon Dev cards; short horizons may need a different width scale (sqrt-of-horizon from daily vol) -
   that is textbook and legal. Any change must be generic (by horizon and panel type, never per card) and must be
   measured on the 22 cards with the same scorer; report in-sample and a split-half check (e.g. odd/even cards).
5. If a variant is clearly better (lower mean on both halves, not driven by one card) build it ON THE VPS, push to
   ghcr (approved kind), verify anonymous pull (HTTP 200), write
   `submissions/2026-10-22/submission_t2_final_<name>.json` (copy of the newscale descriptor, new digest, Final ids)
   and a pack script adapted from `submit_track2_final_newscale.ps1`. Otherwise recommend keeping newscale.
6. REPORT `t2work/REPORT_PS1.md`: per-card table newscale vs variant vs baseline, per-family means, the recall
   caveat, the audit (no per-card values, no outcome lookups, no card ids in code), and the recommendation.
Never handle the Team Key; never upload; no CodaBench actions. Shell: write script files and run them with absolute
paths; never `python -`, never heredocs, never stdin piping (hangs on this Windows machine).
