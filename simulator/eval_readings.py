"""Score House readings in the simulator (71 scored cards, base = t8 + drift x0.5).

  readings.json  the agent's standard prompt (memory forbidden): direction_sd, vol_mult, basis
  recall.json    the measurement prompt (memory allowed): point, lo, hi per ASSET@HORIZON
Usage: eval_readings.py [readings.json] [recall.json]
"""
import json
import pathlib
import sys

import numpy as np

import t2_core as C
import t2_sim as T

truth = C.load_truth()
names = T.card_set(truth, "all")
BASE = {"t_nu": 8.0, "drift": 0.5}
cache = {}
for n in names:
    base, cells = T.base_of(n)
    S, _ = C.m0x_draws(n, BASE)
    cache[n] = (base, S, C.truth_vector(truth, n, cells), C.card_of(n), cells)


def primary(fn, subset=None):
    vals, fam = [], {}
    for n, (base, S, y, card, cells) in cache.items():
        if subset is not None and n not in subset:
            continue
        v = min(C.card_score(card, base, fn(n, S, y, cells), y), C.CLIP)
        vals.append(v)
        fam.setdefault(C.family(n), []).append(v)
    return float(np.mean(vals)), {f: float(np.mean(v)) for f, v in sorted(fam.items())}


def show(label, fn, subset=None):
    p, fam = primary(fn, subset)
    print(f"  {label:34s} {p:.4f}   " + "  ".join(f"{f} {v:.3f}" for f, v in fam.items()))


def load(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


std = load(sys.argv[1] if len(sys.argv) > 1 else "readings.json")
rec = load(sys.argv[2] if len(sys.argv) > 2 else "recall.json")
print("base (no text):")
show("t8 + drift0.5", lambda n, S, y, cells: S)

if std:
    ok = {n for n in names if isinstance(std.get(n), dict) and "assets" in std[n]}
    print(f"\nSTANDARD reading: {len(ok)}/{len(names)} cards answered")
    hits = tot = 0
    for n in ok:
        base, S, y, card, cells = cache[n]
        c = S.mean(axis=0)
        for j, (a, _h) in enumerate(cells):
            e = std[n]["assets"].get(a) or {}
            try:
                d = float(e.get("direction_sd", 0) or 0)
            except (TypeError, ValueError):
                d = 0.0
            if d != 0:
                tot += 1
                hits += int(np.sign(d) == np.sign(y[j] - c[j]))
    print(f"  direction views on {tot} cells, right on {hits} ({hits / max(tot, 1):.0%})")

    def std_fn(k, width):
        def f(n, S, y, cells):
            if n not in ok:
                return S
            c, sd = S.mean(axis=0), S.std(axis=0)
            out = S.copy()
            for j, (a, _h) in enumerate(cells):
                e = std[n]["assets"].get(a) or {}
                try:
                    d = float(e.get("direction_sd", 0) or 0)
                    m = float(e.get("vol_mult", 1) or 1)
                except (TypeError, ValueError):
                    d, m = 0.0, 1.0
                d = max(-1.0, min(1.0, d))
                m = max(0.7, min(1.6, m))
                if m < 1 and e.get("basis") != "established":
                    m = 1.0
                if not width:
                    m = 1.0
                out[:, j] = c[j] + k * d * sd[j] + (S[:, j] - c[j]) * m
            return out
        return f
    for k in (0.0, 0.5, 1.0, 2.0):
        show(f"width + direction x{k}", std_fn(k, True))
    show("direction x1, no width", std_fn(1.0, False))

if rec:
    ok = {n for n in names if isinstance(rec.get(n), dict) and "cells" in rec[n]}
    recog = {n for n in ok if rec[n].get("recognised") is True}
    print(f"\nRECALL reading: {len(ok)}/{len(names)} answered, {len(recog)} recognised")

    def rec_fn(a_w, use_band):
        def f(n, S, y, cells):
            if n not in ok:
                return S
            c, sd = S.mean(axis=0), S.std(axis=0)
            out = S.copy()
            for j, (a, h) in enumerate(cells):
                e = rec[n]["cells"].get(f"{a}@{h}") or {}
                try:
                    pt = float(e["point"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(pt) or abs(pt - c[j]) > 8 * sd[j]:
                    continue                      # a unit slip, not a view
                m = 1.0
                if use_band:
                    try:
                        half = (float(e["hi"]) - float(e["lo"])) / 2
                        if half > 0:
                            m = max(0.4, min(1.6, (half / 1.645) / sd[j]))
                    except (KeyError, TypeError, ValueError):
                        pass
                out[:, j] = c[j] + a_w * (pt - c[j]) + (S[:, j] - c[j]) * m
            return out
        return f
    # accuracy of the recalled point
    err_b, err_r, k_right, k_tot = [], [], 0, 0
    for n in ok:
        base, S, y, card, cells = cache[n]
        c, sd = S.mean(axis=0), S.std(axis=0)
        for j, (a, h) in enumerate(cells):
            e = rec[n]["cells"].get(f"{a}@{h}") or {}
            try:
                pt = float(e["point"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(pt - c[j]) > 8 * sd[j]:
                continue
            err_b.append(abs(y[j] - c[j]) / sd[j]); err_r.append(abs(y[j] - pt) / sd[j])
            k_tot += 1; k_right += int(np.sign(pt - c[j]) == np.sign(y[j] - c[j]))
    print(f"  cells {k_tot}: direction right {k_right / max(k_tot, 1):.0%}; "
          f"median |error| base {np.median(err_b):.2f} sd -> recall {np.median(err_r):.2f} sd")
    for a_w in (0.25, 0.5, 0.75, 1.0):
        show(f"centre {a_w:.2f} to recall, base width", rec_fn(a_w, False))
    for a_w in (0.5, 0.75, 1.0):
        show(f"centre {a_w:.2f} to recall, model band", rec_fn(a_w, True))
    if recog and len(recog) < len(ok):
        print("  recognised cards only / others:")
        show("  recognised, centre 0.75", rec_fn(0.75, False), recog)
        show("  not recognised, centre 0.75", rec_fn(0.75, False), ok - recog)
        show("  not recognised, base", lambda n, S, y, cells: S, ok - recog)
