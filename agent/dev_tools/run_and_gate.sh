#!/bin/bash
# Produce a Track 2 forecast and put it through the real gates.
#
# Output goes to a NATIVE LINUX path, never /mnt/c. On the Windows drive mounted
# into WSL, an ordinary file reports st_nlink=2, and g0 refuses anything with
# more than one link because the bytes might be shared outside the submission
# tree. The agent is fine; the mount is not a valid place to test it.
#   run_and_gate.sh [unit-name]
set -euo pipefail

UNIT_NAME="${1:-t2-EXAMPLE-ust-curve-1m}"
PUBLIC=/mnt/c/Users/Saifuddin/Documents/track2-forecasting-public
AGENT=/mnt/c/Users/Saifuddin/Documents/agenthon-t1/track2_agent
PY=/home/aifuddin/venvs/agenthon/bin/python
UNIT="$PUBLIC/units/$UNIT_NAME"
OUT="$HOME/t2runs/$UNIT_NAME"

# From the card, not from grep: card.toml holds several dates and only one of
# them is the as-of.
ASOF=$("$PY" "$AGENT/dev_tools/card_asof.py" "$UNIT")

rm -rf "$OUT"
mkdir -p "$OUT"
echo "unit: $UNIT_NAME    asof: $ASOF"
echo "out:  $OUT"
echo

"$PY" "$AGENT/agent.py" forecast --unit-dir "$UNIT" --panels "$UNIT/panels" --text "$UNIT/text" --asof "$ASOF" --out "$OUT/forecast.parquet"

echo
"$PY" "$AGENT/dev_tools/gate_check.py" "$UNIT_NAME" "$OUT"
