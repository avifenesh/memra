#!/usr/bin/env bash
# Runs ON a GB10 host: the v1/v2 probe over the Qwen3.8-27B trunk shapes at one token count.
# usage: probe_shapes.sh T TAG
set -u
D=$HOME/.cache/memra-gemm-v2; T=$1; TAG=$2
for s in "17408 5120" "5120 10240" "5120 6144" "6144 5120" "5120 12288" "5120 1024" "5120 17408"; do
  set -- $s
  for c in 0 2 3 5 6; do
    "$D/f8f4v2_probe" "$1" "$2" "$T" 5 1 "$c" 0 2>&1 | grep -v "G=1 \|G=1048576"
  done > "$D/logs/shapes-$TAG-$1x$2.log"
done
echo PROBE-SHAPES-DONE
