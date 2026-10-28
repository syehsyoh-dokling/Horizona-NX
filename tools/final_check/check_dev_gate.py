"""How many original (non-ps1) units would a horizon gate at G business days touch?
A gate changes a cell only if its declared horizon <= G on a non-monthly panel.
usage: check_dev_gate.py <public_repo_dir> <G> [<G> ...]
"""
import pathlib
import sys
import tomllib

PUB = pathlib.Path(sys.argv[1])
gates = [int(g) for g in sys.argv[2:]]
units = []
for unit in sorted((PUB / "units").iterdir()):
    if not (unit / "card.toml").is_file() or unit.name.endswith("-ps1") or "EXAMPLE" in unit.name:
        continue
    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    units.append((unit.name, card["task"].get("split"), [int(h) for h in card["targets"]["horizons"]],
                  card["metadata"].get("target_frequency", "")))
print(f"{len(units)} non-ps1 units; min horizon over all = {min(min(h) for _, _, h, _ in units)}")
for g in gates:
    hit = [(n, s, h, f) for n, s, h, f in units if min(h) <= g]
    val = [x for x in hit if x[1] == "validation"]
    print(f"gate <= {g:2d} BD touches {len(hit)} units ({len(val)} validation):")
    for x in hit:
        print("   ", x)
