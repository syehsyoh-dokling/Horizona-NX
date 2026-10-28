#!/usr/bin/env bash
# Run one qf-forecast image on every -ps1 unit, sequentially (one T2 container at a time).
# usage: vps_run_ps1.sh <image-ref> <out-tag>
# Mount layout: /input = unit dir (card.toml, panels/, text/), /output = per-unit output dir.
# No network, no env overrides: the image runs exactly as the organizers would run it.
set -euo pipefail
IMG="$1"
TAG="$2"
BASE=/home/webmaster/build/t2ps1
IN="$BASE/inputs"
OUT="$BASE/out_$TAG"
mkdir -p "$OUT"
for u in "$IN"/*-ps1; do
  name=$(basename "$u")
  mkdir -p "$OUT/$name"
  docker run --rm --network=none --name "t2ps1_run" \
    -v "$u":/input:ro -v "$OUT/$name":/output \
    "$IMG" > "$OUT/$name/stdout.json" 2> "$OUT/$name/stderr.txt" || echo "FAIL $name"
done
cd "$BASE" && tar -czf "out_$TAG.tar.gz" "out_$TAG"
ls -la "$BASE/out_$TAG.tar.gz"
