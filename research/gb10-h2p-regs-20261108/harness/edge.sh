#!/usr/bin/env bash
# Runs ON a GB10 host: ragged-shape byte compare (T, T_kv not multiples of 32/48/64, fresh primes,
# the served 131k and 260k final chunks). usage: edge.sh TAG KERNELS (first = reference)
set -u
TAG=$1; KS=$2
D=$HOME/.cache/memra-attn3
L=$D/logs/edge-$TAG.log
{
  echo "# $(date -Is) $(hostname) edge KS=$KS"
  for spec in "128 128,129,160,161" "130 130,4226" "176 176,177" "192 192,4097" "1000 1000,1031,33000" \
              "1024 8224,131070" "4033 4033,4097,70001" "4075 4075,8171,131051" "4096 4096,262144" "3567 259951"; do
    set -- $spec
    "$D/probe/probe3" "$1" "$2" 1 "$KS" | grep -E "^T |kernel"
  done
  echo "EDGE-DONE $TAG"
} > "$L" 2>&1
grep -c "mismatch 0 nonfinite 0" "$L"; grep -E "mismatch [1-9]|nonfinite [1-9]" "$L" | head; tail -n 1 "$L"
