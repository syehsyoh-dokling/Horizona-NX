#!/usr/bin/env bash
# Build the newscalegate image on the VPS from the shipped context and run it on the 22 -ps1 units.
set -euo pipefail
BASE=/home/webmaster/build/t2ps1
TAG=ghcr.io/syehsyoh-dokling/qf-forecast:final-newscalegate-20261022
cd "$BASE"
rm -rf ctx_gate && mkdir -p ctx_gate
cp m0_exact_gate.py ctx_gate/m0_exact.py
cp Dockerfile.newscalegate ctx_gate/Dockerfile
sha256sum ctx_gate/m0_exact.py
docker build -t "$TAG" ctx_gate 2>&1 | tail -3
docker image inspect "$TAG" --format 'ID {{.Id}} SIZE {{.Size}}'
docker run --rm --entrypoint sh "$TAG" -c 'sha256sum /agent/forecast_core/m0_exact.py; grep -n "ratio > 2\|widen_gate" /agent/forecast_core/m0_exact.py | head -5; env | grep ^T2_ | sort'
./vps_run_ps1.sh "$TAG" newscalegate 2>&1 | tail -3
df -h / | tail -1
