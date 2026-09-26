#!/usr/bin/env bash
# WP-A day 32, the "both cards" half on the local RTX 5090 Laptop GPU (9B NVFP4 MTP artifact) with the
# day-32 tree's local release binary (the H2D half): the identity gate default ON, the fault gate (twelve
# cells) in the default and the plain arm, then the hit gate OFF and ON (C's day-23 shape on this card: MEMRA_HOSTGATE_CACHE_MB=64 so the
# 54.8 MB seed entries evict one another). Every gate takes its own flock on /tmp/memra-5090.lock (MEMRA_GPU_LOCK).
# The lead's battery shares the card: before each cell wait, bounded (15 x 120 s), until the card carries no compute
# app and reports at least 20000 MiB free, logging each wait with nvidia-smi's own listing; the holder is never
# inspected beyond that listing and never signalled. A cell whose wait runs out is recorded NOT RUN.
# Executed-not-qualified. usage: battery-5090.sh <out_root> <model.gguf> <bin> [identity-rerun]
set -uo pipefail
ROOT=$1; MODEL=$2; BIN=$3
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
until grep -q '^rc=' "$ROOT/build.log" 2>/dev/null; do sleep 20; done
grep -q '^rc=0$' "$ROOT/build.log" || { echo "$(date -u +%FT%TZ) build failed; no cell" | tee -a "$ROOT/battery.log"; exit 1; }
TREE=$(git rev-parse HEAD)
sha256sum "$BIN" > "$ROOT/binary.sha256"
wait_idle() {
    for attempt in $(seq 1 15); do
        apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -z "$apps" ] && [ "${free_mib:-0}" -ge 20000 ]; then return 0; fi
        echo "$(date -u +%FT%TZ) wait $attempt before $1: free=${free_mib}MiB apps=[${apps//$'\n'/; }]" | tee -a "$ROOT/battery.log"
        sleep 120
    done
    return 1
}
cell() { # $1 name $2 env-string $3.. command (the gate; $OUT/ev is its evidence dir)
    local name=$1 envs=$2; shift 2
    local OUT=$ROOT/$name; mkdir -p "$OUT"
    if ! wait_idle "$name"; then
        echo "$(date -u +%FT%TZ) $name NOT RUN: the card never freed in 15 waits" | tee -a "$ROOT/battery.log" "$OUT/NOT-RUN"
        return
    fi
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.before.csv" 2>&1
    echo "$(date -u +%FT%TZ) start $name tree=$TREE env=[$envs]" | tee -a "$ROOT/battery.log"
    # shellcheck disable=SC2086
    env $envs "$@" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
    echo "$rc" > "$OUT/gate.exit"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.after.csv" 2>&1
    {
        echo "cell=$name gate=$1"; echo "env=$envs"; echo "lock=$MEMRA_GPU_LOCK owner=gate-internal-canonical"
        echo "tree=$TREE"; echo "binary_sha256=$(cut -d' ' -f1 "$ROOT/binary.sha256")"; echo "model=$(basename "$MODEL")"
        echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"; echo "status=executed-not-qualified"
    } > "$OUT/CELL.txt"
    echo "$(date -u +%FT%TZ) done $name rc=$rc $(grep -hE 'GATE: ' "$OUT/gate.log" | tail -1)" | tee -a "$ROOT/battery.log"
}
if [ "${4:-}" = identity-rerun ]; then # the one cell whose wait ran out in the first pass, same gate, env and binary
    cell identity-default-on-rerun "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh "$MODEL" "$BIN"
    echo "$(date -u +%FT%TZ) LOCAL-RERUN-DONE" | tee -a "$ROOT/battery.log"; exit 0
fi
cell identity-default-on "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh "$MODEL" "$BIN"
cell fault-default "MEMRA_HOSTGATE_CACHE_MB=64" bash tools/kv-host-contract-fault-gate.sh "$MODEL" "$BIN"
cell fault-plain "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_SERVE_SPEC=0" bash tools/kv-host-contract-fault-gate.sh "$MODEL" "$BIN"
cell hit-off "" bash tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN"
cell hit-on "MEMRA_KV_HOST_CONTRACTS=1" bash tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN"
echo "$(date -u +%FT%TZ) LOCAL-BATTERY-DONE" | tee -a "$ROOT/battery.log"
