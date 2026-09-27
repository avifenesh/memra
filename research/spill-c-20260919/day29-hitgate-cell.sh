#!/usr/bin/env bash
# Day 29 hit gate under the collector's hold (the day-28 owed run): tools/spec-on-cache-hit-gate.sh
# --external-lock FD, door OFF (MEMRA_KV_HOST_CONTRACTS unset) then door ON (MEMRA_KV_HOST_CONTRACTS=1; the gate
# exports MEMRA_KV_HOST_MB=8192 and asserts the door engaged), both arms in ONE collector hold on the target
# card. The gate takes no lock of its own and writes <ev>/LOCK.json (owner collector) before any boot. Verdict
# line, exit code, door-line and restore-line census per arm (the day-27 shape). Executed-not-qualified. No
# host, id or price here. usage: day29-hitgate-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd "$TREE" || exit 1
OUT=$R/hitgate
mkdir -p "$OUT"
sha256sum "$BIN" | tee "$OUT/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "gate=tools/spec-on-cache-hit-gate.sh --external-lock $fd qwen"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "status=executed-not-qualified"
} > "$OUT/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.before.csv" 2>&1
rc=0
for arm in off on; do
    EVD=$OUT/ev-$arm; mkdir -p "$EVD"
    ENVS=()
    [ "$arm" = on ] && ENVS+=("MEMRA_KV_HOST_CONTRACTS=1")
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.$arm.before.csv" 2>&1
    date -u +%FT%TZ > "$OUT/$arm.started.txt"
    env "${ENVS[@]}" bash tools/spec-on-cache-hit-gate.sh --external-lock "$fd" qwen "$MODEL" "$BIN" "$EVD" > "$OUT/$arm.gate.log" 2>&1; arc=$?
    date -u +%FT%TZ > "$OUT/$arm.finished.txt"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.$arm.after.csv" 2>&1
    echo "$arc" > "$OUT/$arm.gate.exit"
    grep -E "GATE: " "$OUT/$arm.gate.log" | tail -1 > "$OUT/$arm.verdict.txt"
    grep -rHE "\[prefix-host\] on: budget|contracts door ON|\[kv-host-contracts\]|(capture|restore|demote|promote) (submitted|published|landed) off the tick|refused \(contracts door\)|restore refused|restore dropped|capture dropped|TIER DISABLED|OFF-TICK DISABLED" "$EVD" 2>/dev/null | sort > "$OUT/$arm.door-lines.txt"
    grep -rHE "restore submitted off the tick|restore landed off the tick|restore refused|restore dropped|RESTORE OFF-TICK DISABLED" "$EVD" 2>/dev/null | sort > "$OUT/$arm.restore-lines.txt"
    echo "hit-$arm rc=$arc $(cat "$OUT/$arm.verdict.txt") lock=$(tr -d '\n' < "$EVD/LOCK.json" 2>/dev/null | cut -c1-200) door_lines=$(wc -l < "$OUT/$arm.door-lines.txt") restore_route_lines=$(wc -l < "$OUT/$arm.restore-lines.txt")" | tee -a "$OUT/summary.txt"
    [ "$arc" -eq 0 ] || rc=1
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.after.csv" 2>&1
echo "hitgate-day29 rc=$rc" | tee "$OUT/exit.txt"
exit $rc
