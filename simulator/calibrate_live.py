"""Calibrate the new-scale simulator against the live per-card scores (13 Oct board). Maps the 15
renamed cards, scores candidate configurations per card, and prints live/sim ratios so the
divisor discrepancy can be seen per card and per family. Usage: calibrate_live.py"""
import pathlib
import numpy as np, pandas as pd
import t2_core as C, t2_sim as T, new_scale as N
RENAME = {"t2-F2-fragile-five-brl-2013": "t2-F2-brl-transfer-2013", "t2-F2-fragile-five-inr-2013": "t2-F2-inr-transfer-2013",
          "t2-F3-cpi-shock-cross-2022": "t2-F3-cpi-release-cross-2022", "t2-F3-fragile-five-joint-2013": "t2-F3-em-transfer-joint-2013",
          "t2-F3-oil-fx-split-2022": "t2-F3-energy-escalation-fx-2022", "t2-F3-bear-flattener-2022": "t2-F3-front-loaded-hikes-2022",
          "t2-F3-funding-flip-2024": "t2-F3-funding-carry-fx-2024", "t2-F3-vaccine-rotation-2020": "t2-F3-readout-calendar-factors-2020",
          "t2-F3-taper-steepener-2013": "t2-F3-taper-testimony-curve-2013", "t2-F4-quant-crowding-2007": "t2-F4-factor-joint-credit-2007",
          "t2-F4-mkt-selloff-2011": "t2-F4-mkt-debt-ceiling-2011", "t2-F4-china-panic-2016": "t2-F4-mkt-new-year-2016",
          "t2-F4-momentum-reversal-2009": "t2-F4-momentum-bank-stress-2009", "t2-F4-qe1-expansion-2009": "t2-F4-ust10y-january-fomc-2009",
          "t2-F4-hike-cycle-2021Q4b": "t2-F4-ust2y-december-fomc-2021"}
live = {l.split()[0]: float(l.split()[1]) for l in open(pathlib.Path(__file__).with_name("live_1013_percard.txt")) if l.strip()}
truth = C.load_truth(); names = T.card_set(truth, "all")
NEWSCALE = {"F1": {"t_nu": 5.0, "drift": "zero"}, "F2": {"widen": 1.5, "drift": "zero"},
            "F3": {"widen": 1.25, "t_nu": 5.0, "drift": "zero"}, "F4": {"widen": 3.0}}
ROOT = pathlib.Path(r"C:/Users/SAIFUD~1/AppData/Local/Temp/claude/C--Users-Saifuddin-Documents-Agenthon-2026/1c8bb465-408f-4455-b5b2-84ff918c0701/scratchpad/t2_recall_run71")
rows = []
for n in names:
    ln = RENAME.get(n, n)
    base, cells = C.org_m0_draws(n); y = C.truth_vector(truth, n, cells)
    s_new, _ = C.m0x_draws(n, NEWSCALE[C.family(n)])
    sim_new = N.score_new(n, s_new, y); sim_m0 = N.score_new(n, base, y)
    f = ROOT / n / "forecast.parquet"; sim_rc = None
    if f.exists():
        df = pd.read_parquet(f); wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
        sim_rc = N.score_new(n, np.column_stack([wide[(a, h)].to_numpy() for a, h in cells]), y)
    rows.append((n, C.family(n), live.get(ln), sim_new, sim_m0, sim_rc))
lv = np.array([r[2] for r in rows]); sn = np.array([r[3] for r in rows]); sm = np.array([r[4] for r in rows]); sr = np.array([r[5] or np.nan for r in rows])
cl = lambda x: np.minimum(x, 8)
print(f"live mean(clip8) {cl(lv).mean():.4f} | sim newscale {cl(sn).mean():.4f} | sim M0 {cl(sm).mean():.4f} | sim recall-opt run {np.nanmean(cl(sr)):.4f}")
for lab, s in (("newscale", sn), ("recall-opt", sr)):
    ok = ~np.isnan(s)
    r = lv[ok] / s[ok]
    print(f"{lab}: corr(log) {np.corrcoef(np.log(lv[ok]), np.log(s[ok]))[0,1]:.3f}  live/sim median {np.median(r):.3f}  IQR {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f}")
print("per-family live/sim-newscale median:", {f: round(float(np.median([r[2] / r[3] for r in rows if r[1] == f])), 3) for f in ("F1", "F2", "F3", "F4")})
print("\ncard  live  sim-newscale  sim-M0  (sorted by |log ratio|)")
for n, f, l, s, m, rc in sorted(rows, key=lambda r: -abs(np.log(r[2] / r[3])))[:14]:
    print(f"  {n:40s} {f} live {l:7.3f} new {s:7.3f} M0 {m:7.3f}")
