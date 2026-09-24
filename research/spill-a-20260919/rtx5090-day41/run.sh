#!/usr/bin/env bash
# WP-A day 41 on the local RTX 5090 (DAY41 section 1's acceptance): the fault gate default and plain on the tip binary
# (every cell, the three kcell cells included), under ONE bounded hold of /tmp/memra-5090.lock (90 x 120 s; then no compute
# app and >= 20000 MiB free, 15 x 60 s), each gate through --external-lock 9. usage: run.sh <out> <model> <bin>
set -uo pipefail
OUT=$1; MODEL=$2; BIN=$3
cd "$(dirname "$0")/../../.." || exit 1
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
exec 9>/tmp/memra-5090.lock
held=0; for a in $(seq 1 90); do flock -w 120 9 && { held=1; break; }; log "hold attempt $a busy"; done
[ $held = 1 ] || { log "NOT RUN: lock"; exit 2; }
idle=0; for a in $(seq 1 15); do apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '); [ -z "$apps" ] && [ "${free:-0}" -ge 20000 ] && { idle=1; break; }; log "idle wait $a apps=[$apps]"; sleep 60; done
[ $idle = 1 ] || { log "NOT RUN: card not idle"; exit 2; }
log "hold taken"
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
for arm in default plain; do
  envs="MEMRA_HOSTGATE_CACHE_MB=64"; [ $arm = plain ] && envs="$envs MEMRA_SERVE_SPEC=0"
  mkdir -p "$OUT/fault-$arm"
  env $envs bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BIN" "$OUT/fault-$arm/ev" > "$OUT/fault-$arm/gate.log" 2>&1
  log "gate fault-$arm rc=$? $(grep -h 'GATE: ' "$OUT/fault-$arm/gate.log" | tail -1)"
done
log "released"
