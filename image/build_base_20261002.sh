#!/usr/bin/env bash
# 2 Oct: T2 "recall" build = drift05 + document direction (F1, F3) + House-model recall (F2, F4,
# as-of <= 2025-06-30). Build on the VPS (memory-capped); gates g0-g3 with the model OFF (4 cards)
# and with the House model ON through the FREE lane (4 cards, one per family); push only if every
# gate passes. Usage: t2r_build.sh NAME
set -u
name=${1:-recall}
B=/home/webmaster/build/t2v1002
PKG=/home/webmaster/build/t2var/pkg
UNITS=/home/webmaster/build/t2adm1001/u/units
REPO=syehsyoh-dokling/qf-forecast
IMG=ghcr.io/$REPO:dev-20261002-$name
MODEL=nvidia/nemotron-3-super-120b-a12b:free
case "$MODEL" in *:free) ;; *) echo "refusing non-free model"; exit 1;; esac
rm -rf "$B" && mkdir -p "$B/ctx"
tar -xzf /home/webmaster/build/t2r_ctx.tar.gz -C "$B/ctx"
cd "$B"
sudo -n env DOCKER_BUILDKIT=0 timeout 900 docker build --memory=2g --memory-swap=2g \
  --build-arg T2_TAILS=8 --build-arg T2_TEXTVOL=1 --build-arg T2_VARIANT=drift=0.5 \
  --build-arg "T2_TEXTMODE=F1:dir;F2:recall;F3:dir;F4:recall" --build-arg T2_RECALL_MAX=2025-06-30 \
  -t "$IMG" ctx > "build_$name.log" 2>&1
rc=$?; echo "=== $name build exit=$rc"; [ $rc -eq 0 ] || { tail -5 "build_$name.log"; exit 1; }
echo "    env: $(sudo -n docker run --rm --entrypoint sh "$IMG" -c 'env | grep -E "T2_|MODEL_M|MODEL_T" | sort | tr "\n" " "')"
LIMITS=(--rm --read-only --user 65534:65534 --memory 2g --memory-swap 2g --cpus 1
        --pids-limit 256 --tmpfs /tmp:rw,size=64m,noexec,nosuid,nodev)
export MODEL_TOKEN
MODEL_TOKEN=$(cat /home/webmaster/.agenthon_openrouter_key)
bad=0
run_one() {  # mode card
  local mode=$1 n=$2 out="$B/out_$1/$2" c="$B/clean_$1/$2"
  mkdir -p "$out" "$c"; chmod 777 "$out"
  if [ "$mode" = off ]; then
    timeout 300 sudo -n docker run "${LIMITS[@]}" --network none -v "$UNITS/$n:/input:ro" -v "$out:/output" "$IMG" \
      forecast --panels /input/ --text /input/text/ --out /output/forecast.parquet > "$B/run_${mode}_$n.log" 2>&1
  else
    timeout 1800 sudo -n --preserve-env=MODEL_TOKEN docker run "${LIMITS[@]}" -e MODEL_TOKEN \
      -e MODEL_ENDPOINT=https://openrouter.ai/api -e MODEL_NAME=$MODEL \
      -v "$UNITS/$n:/input:ro" -v "$out:/output" "$IMG" \
      forecast --panels /input/ --text /input/text/ --out /output/forecast.parquet > "$B/run_${mode}_$n.log" 2>&1
  fi
  local rc=$?
  cp "$out/forecast.parquet" "$out/forecast_meta.json" "$out/forecast_rationale.md" "$c/" 2>/dev/null
  local res
  res=$(sudo -n docker run --rm --network none --memory 1g -e PYTHONPATH=/pkg \
    -v "$PKG:/pkg:ro" -v "$UNITS/$n:/unit:ro" -v "$c:/out:ro" t2dev:latest \
    python -m qfbench2_track_forecasting.scoring score --card /unit/card.toml \
    --forecast /out/forecast.parquet 2>&1)
  local adm=NOT_ADMISSIBLE; echo "$res" | grep -q '"admissible": true' && adm=admissible || bad=1
  echo "    [$mode] $n rc=$rc $adm $(grep -o '"centre_moves".*' "$B/run_${mode}_$n.log" | cut -c1-260)"
}
for n in t2-EXAMPLE-ust-curve-1m t2-F1-liftoff-telegraph-2015 t2-F1-cpi-glidepath-2023 t2-F3-funding-flip-2024; do run_one off $n; done
for n in t2-F1-cad-boc-2017 t2-F2-abenomics-2012 t2-F3-dollar-squeeze-2020 t2-F4-covid-rates-2020; do run_one on $n; done
tar -czf /home/webmaster/build/t2r_out.tar.gz -C "$B" out_on
[ $bad -eq 0 ] || { echo "GATES FAILED - not pushed"; exit 1; }
sudo -n docker push "$IMG" > "push_$name.log" 2>&1
rc=$?
dig=$(sudo -n docker image inspect "$IMG" --format '{{index .RepoDigests 0}}' 2>/dev/null | sed 's/.*@//')
tok=$(curl -s "https://ghcr.io/token?scope=repository:$REPO:pull" | python3 -c "import json,sys; print(json.load(sys.stdin).get('token',''))")
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $tok" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  -H 'Accept: application/vnd.oci.image.manifest.v1+json' "https://ghcr.io/v2/$REPO/manifests/$dig")
echo "$name push_exit=$rc digest=$dig anonymous_http=$code"
echo T2R_DONE
