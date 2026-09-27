#!/usr/bin/env bash
# Day 27 local chain (RTX 5090 Laptop GPU, MEMRA_CTX=65536, the day-26 shape): the after cell, then the park cell's
# four runs (order 1: default then compact; order 2: compact then default). Each run is one server boot through
# run-day26-cell.sh, which takes /tmp/memra-5090.lock by flock (waits up to 3600 s) and runs under the CPU quota.
set -uo pipefail
WT=/home/avifenesh/projects/wt-spill-b; R=$WT/research/spill-b-20260919/rtx5090-day27
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
log "chain start HEAD=$(git rev-parse HEAD) bin=$(sha256sum target/release/memra-server | cut -c1-16)"
run_cell() { local name=$1 order=$2; shift 2; log "$name start"; env MEMRA_CTX=65536 RIGDIR="$R" "$@" bash research/spill-b-20260919/run-day26-cell.sh "$name" "$order" > "$R/$name.launch.log" 2>&1; log "$name exit=$? (cell exit $(cat "$R/$name.exit" 2>/dev/null))"; sleep 5; }
run_cell after-ab AB
run_cell park-o1-default AB
run_cell park-o1-compact AB MEMRA_KV_PARK_COMPACT=1
run_cell park-o2-compact AB MEMRA_KV_PARK_COMPACT=1
run_cell park-o2-default AB
log "chain done"
