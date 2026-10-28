"""Run the real Track 2 admissibility gates against our output.

This is the smoke test that matters. Whether the forecast scores well is sealed
- no realized outcomes ship publicly - but whether it is ADMISSIBLE is fully
checkable here, and inadmissible is worse than badly scored: it is zero with no
diagnostic.
"""

import pathlib
import sys
import tomllib

PUBLIC = pathlib.Path("/mnt/c/Users/Saifuddin/Documents/track2-forecasting-public")
sys.path.insert(0, str(PUBLIC))

UNIT = PUBLIC / "units" / (sys.argv[1] if len(sys.argv) > 1
                           else "t2-EXAMPLE-ust-curve-1m")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                   else "/mnt/c/Users/Saifuddin/Documents/agenthon-t1/track2_agent/out")

card = tomllib.loads((UNIT / "card.toml").read_text(encoding="utf-8"))

print("output directory:")
for path in sorted(OUT.iterdir()):
    print(f"  {path.name:<26} {path.stat().st_size:>9,} bytes")

from scoring.scoring import build_verifier  # noqa: E402

ctx = {"card": card, "output_dir": OUT, "unit_dir": UNIT,
       "realized": None, "ref_scale": None}
verdict = build_verifier(ctx).run(ctx)

print("\nadmissible:", verdict.admissible)
for name, result in verdict.gate_results.items():
    mark = "pass" if result.passed else "FAIL"
    print(f"  {name:<24} {mark}")
    if not result.passed:
        print(f"      {getattr(result, 'detail', '')}")
print("\nscore:", verdict.score, "(None => sealed outcomes, gates still ran)")
print("labels:", getattr(verdict, "labels", None))
