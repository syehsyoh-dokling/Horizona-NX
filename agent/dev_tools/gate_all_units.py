#!/usr/bin/env python3
"""Run the T2 agent + official smoke gates across EVERY public unit.

Closes the listed readiness gap ("belum diuji di semua 104 public units"):
each unit gets an in-process agent run (statistical floor - no model, no
quota) followed by `qfbench2-smoke --track forecasting` on its output. The
product is a per-unit admissibility map: any unit where our survival
contract still leaks becomes a named work item instead of a final-phase
surprise.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

PY = "/home/aifuddin/venvs/agenthon/bin/python"
SMOKE = "/home/aifuddin/venvs/agenthon/bin/qfbench2-smoke"
AGENT = "/mnt/c/Users/Saifuddin/Documents/agenthon-t1/track2_agent/agent.py"
UNITS = pathlib.Path("/mnt/c/Users/Saifuddin/Documents/track2-forecasting-public/units")


def main() -> int:
    units = sorted(p for p in UNITS.iterdir() if p.is_dir())
    rows = []
    for unit in units:
        out = pathlib.Path(tempfile.mkdtemp(prefix="t2gate-"))
        command = [PY, AGENT, "forecast",
                   "--panels", str(unit / "panels"),
                   "--text", str(unit / "text"),
                   "--unit-dir", str(unit),
                   "--out", str(out / "forecast.parquet")]
        if not (unit / "text").is_dir():
            command.append("--no-text")
        agent = subprocess.run(command, capture_output=True, text=True,
                               timeout=300)
        smoke = subprocess.run(
            [SMOKE, str(unit), str(out), "--track", "forecasting"],
            capture_output=True, text=True, timeout=300)
        text = (smoke.stdout or "") + (smoke.stderr or "")
        admissible = "admissible=True" in text
        labels = ""
        if "labels=" in text:
            labels = text.split("labels=")[-1].splitlines()[0][:60]
        rows.append({"unit": unit.name, "admissible": admissible,
                     "agent_rc": agent.returncode, "labels": labels})
        mark = "OK " if admissible else "FAIL"
        print(f"{mark} {unit.name:<44} {labels if not admissible else ''}",
              flush=True)
        subprocess.run(["rm", "-rf", str(out)])

    good = sum(1 for r in rows if r["admissible"])
    print(f"\nadmissible: {good}/{len(rows)}")
    report = pathlib.Path("/home/aifuddin/t2runs/gate_all_units.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"-> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
