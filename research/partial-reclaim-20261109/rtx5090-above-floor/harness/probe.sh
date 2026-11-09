#!/usr/bin/env bash
# The config probe: boot A once per interactive cap and read the derived prefix budget (`[prefix-cache] on:`).
#   gpulock run --ttl 30m --for "memra #1124: 5090 above-floor probe" -- bash probe.sh OUT CAP...
set -u
OUT=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
for cap in "$@"; do
  CELL_A=$CELL_A CELL_B=$CELL_A bash "$HERE/cell.sh" probe "A" "$OUT/cap$cap" "MEMRA_MAX_SESSIONS=$cap"
done
