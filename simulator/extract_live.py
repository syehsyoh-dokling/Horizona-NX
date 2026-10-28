"""Per-card LIVE leaderboard scores of our earlier uploads, recovered from the session transcript
pastes (recovered/user_*.txt). Writes live_scores.json: {label: {"primary": x, "cards": {unit: s}}}."""
import json
import pathlib
import re
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
runs = []
for p in sorted((HERE / "recovered").glob("user_*.txt"), key=lambda p: int(p.stem.split("_")[-1])):
    text = p.read_text(encoding="utf-8")
    for block in text.split("run summary")[1:]:
        m = re.search(r"\n\s*71\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)", block)
        cards = {k: float(v) for k, v in re.findall(r"(t2-F\d-[\w-]+)\s+scored\s+([\d.]+)", block)}
        other = re.findall(r"(t2-F\d-[\w-]+)\s+(?!scored)(\w[\w ]*?)\s", block)
        runs.append({"file": p.name, "reported": float(m.group(4)) if m else None, "n": len(cards),
                     "recomputed": round(st.mean(min(v, 4.0) for v in cards.values()), 4) if cards else None,
                     "cards": cards})
        print(p.name, "reported", runs[-1]["reported"], "cards", len(cards), "recomputed", runs[-1]["recomputed"])

LABELS = {1.2982: "dev-0923 model", 1.2624: "dev-0924 drift-drop", 1.1773: "cal-v3 window504",
          1.2303: "cal-v2 window1260", 1.2509: "cal-v1 house-text", 1.146: "cal-v4 per-family",
          1.1368: "m0b", 1.0: "m0x exact"}
out = {}
for r in runs:
    key = r["reported"] if r["reported"] is not None else r["recomputed"]
    label = LABELS.get(key) or LABELS.get(r["recomputed"]) or f"run {key}"
    if label in out:
        label += f" ({r['file']})"
    out[label] = {"primary_reported": r["reported"], "primary_recomputed": r["recomputed"],
                  "n_cards": r["n"], "source": r["file"], "cards": r["cards"]}
(HERE / "live_scores.json").write_text(json.dumps(out, indent=1))
print({k: (v["primary_reported"], v["primary_recomputed"], v["n_cards"]) for k, v in out.items()})
