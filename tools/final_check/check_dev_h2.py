"""Does the strict-ratio fix change anything on the original 103 practice units (incl. the 71
validation cards the Dev board scored)? It can only matter where ratio == 2.0 exactly: an h=2 cell
from a Mon/Tue/Wed as-of, or an h=1 cell from a Friday as-of, on a business-daily panel.
usage: check_dev_h2.py <public_repo_dir>
"""
import pathlib
import sys
import tomllib

import pandas as pd

PUB = pathlib.Path(sys.argv[1])
hit = []
n = 0
splits = {}
for unit in sorted((PUB / "units").iterdir()):
    if not (unit / "card.toml").is_file() or unit.name.endswith("-ps1") or "EXAMPLE" in unit.name:
        continue
    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    n += 1
    split = card["task"].get("split", "?")
    splits[split] = splits.get(split, 0) + 1
    asof = pd.Timestamp(card["provenance"]["data_cutoff"])
    hs = [int(h) for h in card["targets"]["horizons"]]
    freq = card["metadata"].get("target_frequency", "")
    if 2 in hs or (1 in hs and asof.weekday() == 4) or min(hs) <= 5:
        hit.append((unit.name, split, str(asof.date()), asof.day_name(), hs, freq))
print(f"{n} non-ps1 units, splits {splits}")
print(f"units with any horizon <= 5 BD (candidates for a ratio==2 case): {len(hit)}")
for h in hit:
    print("  ", h)
