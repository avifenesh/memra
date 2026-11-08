#!/usr/bin/env bash
# Laptop: the sm_120a record for #1102's head with the measured --accept-regression reason from the
# quiet decode A/B (decode-ab-out3: ACCEPT, -0.12%, p 0.34). Attempt 3's chain could not start it
# (its out dir name collided with an older run's). Build lock first, then the 5090.
# usage: record-5090.sh HEAD_SHA BASE_SHA OUT
set -u
HEAD=$1; BASE=$2; OUT=$3
G=~/.cache/memra-gemm-v3
cd "$G/eng" || exit 1
flock "$HOME/.cache/memra-rig-build.lock" \
  gpulock run --ttl 60m --for "memra #1102 sm_120a record, accepted perf flag (gemm-v2 agent)" --wait 2h -- \
  python3 "$G/cells-accept.py" "$G/eng" "$HEAD" "$BASE" "$G/cells-5090" "$OUT" "$G/decode-ab-out3/decode-ab.json"
echo "$(TZ=Asia/Jerusalem date +%T) RECORD-5090-DONE rc=$?"
