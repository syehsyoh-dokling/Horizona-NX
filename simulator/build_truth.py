"""Realized outcomes of the public Track 2 cards, for LOCAL evaluation only -> truth.json.

Two independent routes per target asset, both cross-checked against the card's OWN panel on the
dates at or before the as-of before any post-as-of value is trusted:
  sibling : the same panel file in the public unit whose copy runs latest (panels are cut only at
            each card's own as-of; README 'Leakage rules' rule 5). Same builder, same vintage.
  source  : the public source series (FRED H.15 / H.10, Ken French daily factors) with the
            transform that reproduces the card's panel.
Target value of a cell = the h-th row after the as-of (level), or the sum of the next h per-step
returns (log_return panels hold decimal daily returns; `--logsum` hypotheses are kept as alt).
A cell is VERIFIED when its route reproduces the card's own pre-as-of panel exactly and (when both
routes exist) the two routes agree on the target value.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import t2_core as C

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "public_series"
FRED_MAP = {"UST_2Y": "DGS2", "UST_5Y": "DGS5", "UST_7Y": "DGS7", "UST_10Y": "DGS10", "UST_20Y": "DGS20",
            "UST_30Y": "DGS30", "EUR": "DEXUSEU", "GBP": "DEXUSUK", "JPY": "DEXJPUS",
            "CHF": "DEXSZUS", "AUD": "DEXUSAL", "CAD": "DEXCAUS", "NZD": "DEXUSNZ",
            "SEK": "DEXSDUS", "NOK": "DEXNOUS", "DKK": "DEXDNUS", "CNY": "DEXCHUS",
            "INR": "DEXINUS", "BRL": "DEXBZUS"}
KF_MAP = {"MKT": ("ff3_daily", "Mkt-RF"), "SMB": ("ff3_daily", "SMB"), "HML": ("ff3_daily", "HML"),
          "MOM": ("mom_daily", "Mom")}
TRANSFORMS = {"x": lambda x: x, "x/100": lambda x: x / 100.0, "1/x": lambda x: 1.0 / x}
_cache: dict = {}


def fred(sid):
    d = pd.read_csv(SRC / f"{sid}.csv")
    d.columns = ["date", "v"]
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    d = d.dropna()
    return pd.Series(d["v"].to_numpy(float), index=pd.to_datetime(d["date"]))


def kf(name, col):
    lines = (SRC / f"{name}.csv").read_text(errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip()[:8].isdigit())
    hdr = [h.strip() for h in lines[start - 1].split(",")]
    rows = []
    for l in lines[start:]:
        p = [x.strip() for x in l.split(",")]
        if not p[0].isdigit() or len(p[0]) != 8:
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=hdr)
    idx = pd.to_datetime(df.iloc[:, 0], format="%Y%m%d")
    c = [h for h in df.columns if h.lower().startswith(col.lower())][0]
    return pd.Series(pd.to_numeric(df[c]).to_numpy(float), index=idx)


def source(asset):
    if asset not in _cache:
        if asset in FRED_MAP and (SRC / f"{FRED_MAP[asset]}.csv").exists():
            _cache[asset] = (fred(FRED_MAP[asset]), f"FRED {FRED_MAP[asset]}")
        elif asset in KF_MAP:
            _cache[asset] = (kf(*KF_MAP[asset]), f"Ken French {KF_MAP[asset][0]}:{KF_MAP[asset][1]}")
        else:
            _cache[asset] = (None, None)
    return _cache[asset]


def match(panel: pd.Series, src: pd.Series):
    """Best transform of src onto the card's panel: (name, max abs err, rel err, n common, n panel,
    n panel dates missing from the source)."""
    j = panel.to_frame("p").join(src.rename("s"), how="inner")
    if len(j) < 20:
        return None
    best = None
    for name, f in TRANSFORMS.items():
        with np.errstate(all="ignore"):
            err = float(np.nanmax(np.abs(f(j["s"].to_numpy()) - j["p"].to_numpy())))
        scale = float(np.nanmax(np.abs(j["p"].to_numpy()))) or 1.0
        if best is None or err < best[1]:
            best = (name, err, err / scale, len(j), len(panel), len(panel) - len(j))
    return best


def sibling_table():
    """{(panel stem, asset): (unit, series)} from the unit whose copy of that panel ends latest."""
    out = {}
    for name in C.unit_names():
        for p in sorted((C.UNITS / name).glob("*.parquet")):
            df = pd.read_parquet(p)
            col = "asset" if "asset" in df.columns else "asset_id"
            df["date"] = pd.to_datetime(df["date"])
            for a, g in df.groupby(col):
                key = (p.stem, str(a))
                out.setdefault(key, []).append((g["date"].max(), name, p))
    return out


def panel_stem(name, asset):
    for p in sorted((C.UNITS / name).glob("*.parquet"), key=lambda p: p.name):
        df = pd.read_parquet(p)
        col = "asset" if "asset" in df.columns else "asset_id"
        if (df[col].astype(str) == asset).any():
            return p.stem
    return None


# Mixed-calendar card (rates + FX panels, whose holiday rows differ inside the horizon): the live m0b
# per-card score (1.7440) is reproduced only with these COMMON target dates (unique solution of a
# brute-force over +-2 weeks, probe_divergence2.py); own h-th rows give 1.6974.
DATE_OVERRIDE = {"t2-F3-divergence-2014": {63: "2015-01-02", 126: "2015-04-06"}}


def at_date(series: pd.Series, date: str):
    s = series[series.index <= pd.Timestamp(date)]
    return float(s.iloc[-1]), str(s.index[-1].date())


def after(series: pd.Series, asof, h, ttype, logsum=True):
    """(value, target_date) under the h-th-row rule, or None if the series is too short / gapped."""
    fut = series[series.index > asof]
    if len(fut) < h:
        return None
    seg = fut.iloc[:h]
    gaps = np.diff(np.concatenate([[np.datetime64(asof)], seg.index.values])).astype("timedelta64[D]").astype(int)
    if gaps.max() > 7:                      # a withheld stretch (transfer cards), not a daily run
        return None
    if ttype == "log_return":
        x = seg.to_numpy()
        return (float(np.sum(np.log1p(x))) if logsum else float(np.sum(x))), str(seg.index[-1].date())
    return float(seg.iloc[-1]), str(seg.index[-1].date())


def bday_value(series, asof, h, ttype):
    td = asof + pd.offsets.BDay(h)
    if series.index.max() < td:
        return None
    if ttype == "log_return":
        seg = series[(series.index > asof) & (series.index <= td)]
        return float(np.sum(np.log1p(seg.to_numpy())))
    seg = series[series.index <= td]
    return float(seg.iloc[-1])


def main():
    sib = sibling_table()
    out = {}
    for name in C.unit_names():
        card = C.card_of(name)
        asof = pd.Timestamp(card["provenance"]["data_cutoff"])
        ttype = str(card["targets"].get("target_type", "level"))
        freq = str(card["targets"].get("target_frequency", "daily"))
        assets = sorted(str(a) for a in card["targets"]["asset_ids"])
        hs = sorted(int(h) for h in card["targets"]["horizons"])
        rec = {"split": card["task"].get("split"), "family": C.family(name), "asof": str(asof.date()),
               "target_type": ttype, "target_frequency": freq, "cells": [], "assets": {}}
        if freq == "monthly":
            rec["verified"] = False
            rec["note"] = ("monthly macro card (as-of ALFRED vintage panel, release-date target): "
                           "not rebuilt; not a validation card")
            for a in assets:
                for h in hs:
                    rec["cells"].append({"asset": a, "horizon": h, "value": None, "source": None,
                                         "verified": False})
            out[name] = rec
            continue
        for a in assets:
            own = C.panel_series(name, a)
            own = own[own.index <= asof]
            stem = panel_stem(name, a)
            info = {"panel": stem}
            # --- sibling route
            s_series = None
            cands = sorted(sib.get((stem, a), []), reverse=True)
            for end, unit, path in cands:
                if unit == name:
                    continue
                df = pd.read_parquet(path)
                col = "asset" if "asset" in df.columns else "asset_id"
                g = df[df[col].astype(str) == a]
                s = pd.Series(g["value"].astype(float).to_numpy(), index=pd.to_datetime(g["date"]).to_numpy()).sort_index()
                if after(s, asof, max(hs), ttype) is None:
                    continue
                # cross-check on the card's own trailing window (same dates, same values)
                lo = own.index[-min(len(own), 600)]
                a_own, a_sib = own[own.index >= lo], s[(s.index >= lo) & (s.index <= asof)]
                same_dates = a_own.index.equals(a_sib.index)
                err = float(np.max(np.abs(a_own.to_numpy() - a_sib.to_numpy()))) if same_dates else None
                info["sibling"] = {"unit": unit, "same_dates": bool(same_dates), "max_err": err,
                                   "n_checked": int(len(a_own))}
                if same_dates and err is not None and err <= 1e-12:
                    s_series = s
                break
            # --- public-source route
            src, src_name = source(a)
            m = match(own, src) if src is not None else None
            p_series, src_exact = None, False
            if m is not None:
                info["source"] = {"series": src_name, "transform": m[0], "max_err": m[1],
                                  "n_common": m[3], "n_panel": m[4], "panel_dates_not_in_source": m[5]}
                src_exact = bool(m[2] <= 1e-9 and m[5] == 0)
                approx = bool(a in KF_MAP and m[1] <= 0.002)
                if src_exact or approx:
                    p_series = TRANSFORMS[m[0]](src)
                info["source"]["exact"] = src_exact
            rec["assets"][a] = info
            for h in hs:
                cell = {"asset": a, "horizon": h, "value": None, "target_date": None, "source": None,
                        "verified": False, "alt": {}}
                sv = after(s_series, asof, h, ttype) if s_series is not None else None
                pv = after(p_series, asof, h, ttype) if p_series is not None else None
                if sv is not None:
                    cell.update(value=sv[0], target_date=sv[1], verified=True,
                                source=f"sibling panel {info['sibling']['unit']}/{stem}.parquet")
                    if pv is not None:
                        cell["alt"]["public_source"] = pv[0]
                        cell["alt"]["public_source_agrees"] = bool(abs(pv[0] - sv[0]) <= 1e-9 and pv[1] == sv[1])
                        if src_exact and not cell["alt"]["public_source_agrees"]:
                            cell["verified"] = False
                            cell["note"] = "sibling panel and exact public source disagree"
                elif pv is not None:
                    cell.update(value=pv[0], target_date=pv[1], verified=src_exact,
                                source=f"{src_name} [{m[0]}]" + ("" if src_exact else " (approx: revised vintage)"))
                best = s_series if s_series is not None else p_series
                if best is not None:
                    if ttype == "log_return":
                        ls = after(best, asof, h, ttype, logsum=False)
                        cell["alt"]["plain_sum"] = ls[0] if ls else None
                    cell["alt"]["bday_rule"] = bday_value(best, asof, h, ttype)
                    if name in DATE_OVERRIDE and cell["value"] is not None:
                        cell["alt"]["own_hth_row"] = cell["value"]
                        v, d = at_date(best, DATE_OVERRIDE[name][h])
                        cell.update(value=v, target_date=d)
                        cell["source"] += " (common target date identified from the live m0b score)"
                rec["cells"].append(cell)
        vals = [c["value"] for c in rec["cells"]]
        rec["has_truth"] = all(v is not None for v in vals)
        rec["verified"] = bool(rec["has_truth"] and all(c["verified"] for c in rec["cells"]))
        # mixed-calendar cards: the own-row target dates differ between assets at the same horizon
        for h in hs:
            ds = {c["target_date"] for c in rec["cells"] if c["horizon"] == h}
            if len(ds) > 1:
                rec["mixed_calendar"] = True
                if rec["split"] != "validation":
                    rec["verified"] = False
                    rec["note"] = ("assets sit on different holiday calendars inside the horizon; the "
                                   "organizers' common-target-date rule is unknown and there is no live "
                                   "score to pin it")
        out[name] = rec

    (HERE / "truth.json").write_text(json.dumps(out, indent=1))
    df = pd.DataFrame([{"unit": n, "split": r["split"], "fam": r["family"], "has": r.get("has_truth", False),
                        "ver": r["verified"],
                        "src": ";".join(sorted({(c["source"] or "none").split(" ")[0] for c in r["cells"]}))}
                       for n, r in out.items()])
    print(df.groupby(["split", "has", "ver"]).size())
    print(df.groupby(["split", "src"]).size())
    for n, r in out.items():
        if not r["verified"]:
            print("NOT VERIFIED", n, r.get("note", ""), json.dumps(r["assets"])[:400])
        for c in r["cells"]:
            if c.get("alt", {}).get("public_source_agrees") is False:
                print("  disagree", n, c["asset"], c["horizon"], c["value"], c["alt"]["public_source"])


if __name__ == "__main__":
    main()
