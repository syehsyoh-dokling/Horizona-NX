"""Markdown tables for REPORT_PS1.md from compare2_ps1.json (+ container run JSONs).
usage: make_report_tables.py <compare2_ps1.json> <runs_newscale.json> <runs_newscalefix.json> <out_md>
"""
import json
import pathlib
import sys

import numpy as np

cmp_ = json.load(open(sys.argv[1]))
runs_ns = {r["unit"]: r for r in json.load(open(sys.argv[2]))}
runs_fx = {r["unit"]: r for r in json.load(open(sys.argv[3]))}
out = pathlib.Path(sys.argv[4])
cards = cmp_["cards"]
L = []

L.append("### Per-card scores (22 short-horizon practice cards)\n")
L.append("Raw = the official scorer's composite in the card's own units (`raw_unrankable`); scaled = raw components divided by M0's expected error (docs/M0-BASELINE.md section 5, declared-horizon steps), clip 8 — the leaderboard's quantity. `newscale` = the Final entry as its image runs (VPS container output); `newscalefix` = the same image with the one-line section-3.7 fix (VPS container output); `M0` = the organizers' text-blind reference row (exact M0, fixed engine); `RW` = fix + plain random walk (no family widening) for comparison.\n")
L.append("| card | fam | panel | horizons | raw newscale | raw newscalefix | raw M0 | scaled newscale | scaled newscalefix | scaled M0 | scaled RW |")
L.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
for c in cards:
    u = c["unit"]
    s = c["scores"]
    r = c["raw"]
    flag = " **(h=2)**" if c["has_h2"] else ""
    L.append(f"| `{u}`{flag} | {c['family']} | {c['ptype']} | {c['horizons']} | {runs_ns[u]['raw']:.6f} | {runs_fx[u]['raw']:.6f} | {r['m0_fixed']:.6f} | "
             f"{s['newscale_asis']:.4f} | {s['newscale_fixed']:.4f} | {s['m0_fixed']:.4f} | {s['rw_fixed']:.4f} |")
names = ["newscale_asis", "newscale_fixed", "m0_fixed", "rw_fixed", "rw_t8_fixed", "ns_x0.7_fixed"]
mat = np.array([[c["scores"][n] for n in names] for c in cards])
L.append("")
L.append("### Means (scaled, clip 8)\n")
L.append("| group | n | newscale (as is) | newscalefix | M0 reference | RW (fix, no widening) | RW t8 | newscale x0.7 (fitted) |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
def row(label, idx):
    m = mat[idx].mean(axis=0)
    L.append(f"| {label} | {len(idx)} | " + " | ".join(f"{v:.4f}" for v in m) + " |")
row("**all 22**", list(range(len(cards))))
row("odd cards (1,3,5,...)", list(range(0, len(cards), 2)))
row("even cards (2,4,6,...)", list(range(1, len(cards), 2)))
for fam in ("F1", "F2", "F3", "F4"):
    row(f"family {fam}", [i for i, c in enumerate(cards) if c["family"] == fam])
for pt, lab in (("rates", "UST bills (rates_daily)"), ("em_fx", "EM FX (em_fx_ecb_daily)"), ("g10_fx", "G10 FX (g10_fx_daily)")):
    row(lab, [i for i, c in enumerate(cards) if c["ptype"] == pt])
row("cards with an h=2 cell", [i for i, c in enumerate(cards) if c["has_h2"]])
row("cards without h=2", [i for i, c in enumerate(cards) if not c["has_h2"]])
L.append("")
L.append("Raw-composite means per family are in the cards' own units (KRW/USD ~1000 vs EUR/USD ~1.1 vs bills in % p.a.) and are not comparable across cards; they are listed only because the brief asks for them:\n")
L.append("| family | raw newscale | raw newscalefix | raw M0 |")
L.append("|---|---:|---:|---:|")
for fam in ("F1", "F2", "F3", "F4"):
    idx = [c for c in cards if c["family"] == fam]
    L.append(f"| {fam} (n={len(idx)}) | {np.mean([runs_ns[c['unit']]['raw'] for c in idx]):.5f} | {np.mean([runs_fx[c['unit']]['raw'] for c in idx]):.5f} | {np.mean([c['raw']['m0_fixed'] for c in idx]):.5f} |")
L.append("")
L.append("### Candidate vs the current entry (scaled, clip 8)\n")
L.append("| candidate | mean diff | odd half | even half | cards better / worse | leave-one-out range of mean diff | bootstrap P(better) | largest single-card gain / loss |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for ref, title in (("newscale_asis", "vs newscale (as is)"), ("newscale_fixed", "vs newscalefix")):
    L.append(f"| **{title}** | | | | | | | |")
    for n, s in cmp_["summary"][ref].items():
        if n in ("m0_asis",):
            continue
        L.append(f"| {n} | {s['mean_diff']:+.4f} | {s['odd_diff']:+.4f} | {s['even_diff']:+.4f} | {s['better']} / {s['worse']} | [{s['loo_min']:+.4f}, {s['loo_max']:+.4f}] | {s['p_boot_better']:.2f} | {s['largest_single_card_gain']:+.3f} / {s['largest_single_card_loss']:+.3f} |")
L.append("")
L.append("### Width scan (fixed engine, zero drift, normal innovations; mean scaled score, clip 8)\n")
ws = sorted(cards[0]["widen_scan"], key=float)
L.append("| group | " + " | ".join(f"x{w}" for w in ws) + " |")
L.append("|---|" + "---:|" * len(ws))
for key, lab in (("all", "all 22"), ("F1", "F1"), ("F2", "F2"), ("F3", "F3"), ("F4", "F4"), ("rates", "UST bills"), ("em_fx", "EM FX"), ("g10_fx", "G10 FX")):
    idx = list(range(len(cards))) if key == "all" else [i for i, c in enumerate(cards) if c["family"] == key or c["ptype"] == key]
    vals = [np.mean([cards[i]["widen_scan"][w] for i in idx]) for w in ws]
    best = min(range(len(ws)), key=lambda k: vals[k])
    L.append(f"| {lab} (n={len(idx)}) | " + " | ".join(("**%.3f**" if k == best else "%.3f") % v for k, v in enumerate(vals)) + " |")
L.append("")
L.append("### Width calibration of M0's own distribution at 1-4 BD (z = realised minus M0 mean, in M0 sd units)\n")
L.append("| group | cells | mean abs z (normal: 0.798) | rms z (1.0) | inside 50 % band (0.50) | inside 90 % band (0.90) |")
L.append("|---|---:|---:|---:|---:|---:|")
for g, d in cmp_["calibration"].items():
    L.append(f"| {g} | {d['n']} | {d['mean_abs_z']:.3f} | {d['rms_z']:.3f} | {d['in50']:.2f} | {d['in90']:.2f} |")
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote", out, len(L), "lines")
