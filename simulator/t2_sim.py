"""Local leaderboard simulator for Agenthon 2026 Track 2 (Development board, 71 validation cards).

  per-card score = weights . (our component / organizers' M0 component)   [marginal CRPS, variogram, pinball tail;
                   single-cell cards reweighted to (0.714, 0, 0.286)]
  primary        = mean over cards of min(card score, 4.0)                [the board shows it with a minus sign]

Truth comes from truth.json (build_truth.py). TEXT WIDTH IS NOT SIMULATED: these are text-off numbers.

Usage (variant = JSON or the agent's T2_M0_VARIANT syntax, e.g.  t_nu=8,drift=0.5  /  t_nu=8,ewma=0.97:0.5):
  t2_sim.py score [--engine m0x|m0b] [--variant SPEC] [--set verified|all] [--cards]
  t2_sim.py validate      live per-card comparison (m0b, exact M0); finalises the verified flags in truth.json
  t2_sim.py table         exact M0 + the five 1-Oct variants: overall, families, split-half -> results.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

import numpy as np

import t2_core as C

HERE = pathlib.Path(__file__).resolve().parent
FAMS = ("F1", "F2", "F3", "F4")
TABLE = [("exact M0", {}),
         ("t8", {"t_nu": 8}),
         ("t5", {"t_nu": 5}),
         ("t8 + drift x0.5", {"t_nu": 8, "drift": 0.5}),
         ("t8 + ewma(0.97, 0.5)", {"t_nu": 8, "ewma": [0.97, 0.5]}),
         ("t8 + widen 0.95", {"t_nu": 8, "widen": 0.95})]


def card_set(truth: dict, which: str) -> list[str]:
    names = [n for n in C.unit_names("validation") if truth.get(n, {}).get("has_truth")]
    if which == "verified":
        names = [n for n in names if truth[n]["verified"]]
    return names


_BASE: dict = {}


def base_of(name):
    if name not in _BASE:
        _BASE[name] = C.org_m0_draws(name)
    return _BASE[name]


def simulate(variant: dict | None, engine: str, names: list[str], truth: dict) -> dict[str, float]:
    """{card: unclipped leaderboard score} of an engine/variant on the given cards."""
    out = {}
    for n in names:
        base, cells = base_of(n)
        y = C.truth_vector(truth, n, cells)
        if engine == "m0b":
            S = C.m0b_draws(n, cells)
        else:
            S, cells2 = C.m0x_draws(n, variant)
            assert cells2 == cells
        out[n] = C.card_score(C.card_of(n), base, S, y)
    return out


def summarise(scores: dict[str, float]) -> dict:
    names = sorted(scores)
    v = [min(scores[n], C.CLIP) for n in names]
    fam = {f: [min(scores[n], C.CLIP) for n in names if C.family(n) == f] for f in FAMS}
    return {"n": len(v), "primary": st.mean(v), "odd": st.mean(v[0::2]), "even": st.mean(v[1::2]),
            "cards_below_1": sum(x < 1 - 1e-9 for x in v), "clipped": sum(scores[n] > C.CLIP for n in names),
            **{f: (st.mean(x) if x else float("nan")) for f, x in fam.items()},
            **{f"n_{f}": len(x) for f, x in fam.items()}}


def fmt_row(label, s):
    return (f"| {label:<22} | {s['primary']:.4f} | " + " | ".join(f"{s[f]:.4f}" for f in FAMS)
            + f" | {s['odd']:.4f} | {s['even']:.4f} | {s['cards_below_1']}/{s['n']} |")


HEAD = ("| variant                | primary | F1 | F2 | F3 | F4 | odd half | even half | cards<1 |\n"
        "|---|---|---|---|---|---|---|---|---|")


# ------------------------------------------------------------------ validate
def cmd_validate(_args):
    truth = C.load_truth()
    live = json.loads((HERE / "live_scores.json").read_text())
    names = card_set(truth, "all")
    lines = ["# Validation against live per-card leaderboard scores", ""]
    lines.append("Aggregation rule (primary = mean over 71 cards of min(score, 4.0)) applied to the pasted "
                 "per-card live scores of each upload:")
    lines += ["", "| upload | reported | recomputed | cards |", "|---|---|---|---|"]
    for k, r in live.items():
        lines.append(f"| {k} | {r['primary_reported']:.4f} | {r['primary_recomputed']:.4f} | {r['n_cards']} |")

    # m0b
    lm = live["m0b"]["cards"]
    loc = simulate(None, "m0b", names, truth)
    err = {n: min(loc[n], C.CLIP) - lm[n] for n in names}
    e = np.abs(np.array(list(err.values())))
    lv = np.array([lm[n] for n in names])
    lc = np.array([min(loc[n], C.CLIP) for n in names])
    lines += ["", "## m0b upload (vendored reference CLI walk, F1/F3 scale from the last 504 changes; live 1.1368)", "",
              f"- cards simulated: {len(names)} of 71",
              f"- simulated primary **{lc.mean():.4f}** vs live **{lv.mean():.4f}** (old simulator: 1.1327 on 68 cards + 3 live fills)",
              f"- per card |local - live|: exact to the 4 printed decimals (<0.0001) on {int(np.sum(e < 1e-4))}/{len(names)}, "
              f"<0.002 on {int(np.sum(e < 2e-3))}, median {np.median(e):.5f}, max {e.max():.4f}, "
              f"correlation {np.corrcoef(lv, lc)[0, 1]:.6f} (old simulator: 0.998)",
              "- family means live / local: " + ", ".join(
                  f"{f} {np.mean([lm[n] for n in names if C.family(n) == f]):.4f} / "
                  f"{np.mean([min(loc[n], C.CLIP) for n in names if C.family(n) == f]):.4f}" for f in FAMS),
              "- cards not exact:"]
    for n in sorted(names, key=lambda n: -abs(err[n])):
        if abs(err[n]) >= 1e-4:
            lines.append(f"  - {n}: live {lm[n]:.4f}, local {min(loc[n], C.CLIP):.4f} "
                         f"({'data-verified' if truth[n].get('data_verified', truth[n]['verified']) else 'approximate truth'})")

    # exact M0
    lx = live["m0x exact"]["cards"]
    locx = simulate({}, "m0x", names, truth)
    ex = np.abs(np.array([locx[n] - lx[n] for n in names]))
    lines += ["", "## exact M0 upload (m0_exact, no variant; live 1.0000)", "",
              f"- simulated primary **{np.mean([min(locx[n], C.CLIP) for n in names]):.4f}** vs live 1.0000; "
              f"per card max |local - live| {ex.max():.5f}",
              "- live cards not at 1.0000: " + ", ".join(f"{n} live {v:.4f} / local {locx[n]:.4f}"
                                                         for n, v in lx.items() if v != 1.0),
              "  (the organizers' M0 uses the card's [targets] list order of cells on these two cards; our "
              "agent draws in sorted order)"]

    # hypotheses rejected (from the first-pass validation.json, kept for the record)
    vj = HERE / "validation.json"
    if vj.exists():
        rep = json.loads(vj.read_text())["report"]["m0b"]
        lines += ["", "## Hypotheses tested on m0b (first pass, validate.py; exact = |err| < 0.0002 of 71)", "",
                  "| hypothesis | primary | exact | corr |", "|---|---|---|---|"]
        lab = {"value": "h-th row, log_return = sum(r)", "logsum": "h-th row, log_return = sum(log(1+r))  [adopted]",
               "bday_rule": "target = as-of + h business days", "cov_tail": "coverage tail instead of pinball",
               "comp_ratio": "composite ratio instead of per-component ratios"}
        for k, l in lab.items():
            if k in rep:
                lines.append(f"| {l} | {rep[k]['mean_local']:.4f} | {rep[k]['exact(<0.0002)']} | {rep[k]['corr']:.4f} |")

    # finalise flags
    for n in C.unit_names():
        t = truth[n]
        t.setdefault("data_verified", t["verified"])
        if n in err:
            t["live_m0b"] = lm[n]
            t["local_m0b"] = round(min(loc[n], C.CLIP), 6)
            t["live_match"] = bool(abs(err[n]) < 2e-3)
            t["verified"] = bool(t["data_verified"] and t["live_match"])
        else:
            t["verified"] = bool(t["data_verified"])
    (HERE / "truth.json").write_text(json.dumps(truth, indent=1))
    val = [n for n in names]
    lines += ["", "## Truth coverage", "",
              f"- validation cards with truth: {len(val)}/71; data-verified (source reproduces the card's own "
              f"pre-as-of panel exactly): {sum(truth[n]['data_verified'] for n in val)}; live m0b reproduced "
              f"(<0.002): {sum(truth[n]['live_match'] for n in val)}; **verified (both): "
              f"{sum(truth[n]['verified'] for n in val)}**",
              f"- public-dev cards (not on the board, no live check): {sum(1 for n in C.unit_names('public-dev'))} "
              f"cards, truth for {sum(bool(truth[n].get('has_truth')) for n in C.unit_names('public-dev'))}, "
              f"data-verified {sum(truth[n]['verified'] for n in C.unit_names('public-dev'))}"]
    not_v = [n for n in C.unit_names() if not truth[n]["verified"]]
    lines.append("- not verified: " + "; ".join(f"{n} ({truth[n].get('note', 'approximate source / no live match')[:70]})" for n in not_v))
    text = "\n".join(lines) + "\n"
    (HERE / "validation_report.md").write_text(text, encoding="utf-8")
    print(text)


# ------------------------------------------------------------------ score / table
def cmd_score(args):
    truth = C.load_truth()
    names = card_set(truth, args.set)
    variant = C.parse_variant(args.variant)
    scores = simulate(variant, args.engine, names, truth)
    s = summarise(scores)
    print(f"engine {args.engine} variant {variant} on {s['n']} {args.set} validation cards (text off)")
    print(HEAD)
    print(fmt_row(args.engine if args.engine == "m0b" else json.dumps(variant), s))
    if args.cards:
        for n in sorted(scores):
            print(f"  {n:<40} {scores[n]:.4f}")


def cmd_table(_args):
    truth = C.load_truth()
    res = {}
    lines = ["# Variant table (text off; score vs the organizers' M0; lower is better)", ""]
    for which in ("verified", "all"):
        names = card_set(truth, which)
        lines += [f"## {len(names)} validation cards, truth set = {which}", "", HEAD]
        res[which] = {}
        for label, v in TABLE:
            scores = simulate(v, "m0x", names, truth)
            s = summarise(scores)
            res[which][label] = {"variant": v, "summary": s, "cards": scores}
            lines.append(fmt_row(label, s))
        s = summarise(simulate(None, "m0b", names, truth))
        res[which]["m0b (validation engine)"] = {"summary": s}
        lines.append(fmt_row("m0b (live 1.1368)", s))
        lines.append("")
    s = res["verified"]["exact M0"]["summary"]
    lines.append(f"Family sizes ({'verified'}): " + ", ".join(f"{f} {s['n_' + f]}" for f in FAMS)
                 + ". Split halves = odd/even cards in sorted unit-name order.")
    # paired differences vs t8 on the 'all' set
    lines += ["", "## One-change effects vs t8 (all cards; paired per-card differences)", "",
              "| variant | mean diff | cards better | cards worse | sd of diff | t |", "|---|---|---|---|---|---|"]
    t8 = res["all"]["t8"]["cards"]
    for label, _v in TABLE[2:]:
        d = np.array([min(res["all"][label]["cards"][n], C.CLIP) - min(t8[n], C.CLIP) for n in sorted(t8)])
        nz = d[np.abs(d) > 1e-9]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
        lines.append(f"| {label} | {d.mean():+.4f} | {int((nz < 0).sum())} | {int((nz > 0).sum())} | {d.std(ddof=1):.4f} | {t:+.2f} |")
    text = "\n".join(lines) + "\n"
    (HERE / "variant_table.md").write_text(text, encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--engine", choices=("m0x", "m0b"), default="m0x")
    p.add_argument("--variant", default="")
    p.add_argument("--set", choices=("verified", "all"), default="verified")
    p.add_argument("--cards", action="store_true")
    p.set_defaults(fn=cmd_score)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    sub.add_parser("table").set_defaults(fn=cmd_table)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
