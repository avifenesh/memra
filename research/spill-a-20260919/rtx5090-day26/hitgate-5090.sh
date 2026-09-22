#!/usr/bin/env bash
# WP-A day 26, ruling 36's "both cards": the hit gate OFF then ON on the local RTX 5090 Laptop GPU (9B NVFP4 MTP
# artifact) with the day-26 tree's release binary, under the gate's own flock on /tmp/memra-5090.lock
# (MEMRA_GPU_LOCK). The lead and lane C use the card today: before each arm wait, bounded (15 x 120 s), until the
# card carries no compute app and reports at least 20000 MiB free (C's day-26 battery rule), logging each wait with
# nvidia-smi's own listing; the holder is never inspected beyond that listing and never signalled. If the card never
# frees, the arm is recorded NOT RUN. Executed-not-qualified. usage: hitgate-5090.sh <out_root> <model.gguf> <bin>
set -uo pipefail
ROOT=$1; MODEL=$2; BIN=$3
HERE=$(cd "$(dirname "$0")/../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
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
for arm in off on; do
    OUT=$ROOT/hit-$arm; mkdir -p "$OUT"
    if ! wait_idle "hit-$arm"; then
        echo "$(date -u +%FT%TZ) hit-$arm NOT RUN: the card never freed in 15 waits" | tee -a "$ROOT/battery.log" "$OUT/NOT-RUN"
        continue
    fi
    extra=(); [ $arm = on ] && extra=(MEMRA_KV_HOST_CONTRACTS=1)
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.before.csv" 2>&1
    echo "$(date -u +%FT%TZ) start hit-$arm tree=$TREE" | tee -a "$ROOT/battery.log"
    env "${extra[@]}" bash tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
    echo "$rc" > "$OUT/gate.exit"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.after.csv" 2>&1
    {
        echo "cell=hit-$arm gate=tools/spec-on-cache-hit-gate.sh"; echo "env=${extra[*]:-}"; echo "lock=$MEMRA_GPU_LOCK owner=gate-internal-canonical"
        echo "tree=$TREE"; echo "binary_sha256=$(cut -d' ' -f1 "$ROOT/binary.sha256")"; echo "model=$(basename "$MODEL")"
        echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"; echo "status=executed-not-qualified"
    } > "$OUT/CELL.txt"
    echo "$(date -u +%FT%TZ) done hit-$arm rc=$rc $(grep -h 'SPEC-ON-CACHE-HIT GATE' "$OUT/gate.log" | tail -1)" | tee -a "$ROOT/battery.log"
done
echo "LOCAL-HITGATE-DONE" | tee -a "$ROOT/battery.log"
