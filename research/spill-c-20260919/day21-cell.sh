#!/usr/bin/env bash
# day21-cell.sh <cell> <model.gguf> <server_bin> <out_root> [cache_mb] [lock_fd]
# One host-tier gate under one named environment. Without a lock fd the gate takes the canonical GPU
# lock itself (`flock -n`, MEMRA_GPU_LOCK, default the 5090 lock); a busy lock is retried 15 x 120 s
# and the holder is never signalled. With a lock fd (the collector's `--external-lock`, the fd
# replacing @COLLECTOR_LOCK_FD@) the gate inherits the collector's hold. Records
# nvidia-smi compute apps before and after, the binary digest, the tree SHA, the verdict line and
# the exit code. Every cell is executed-not-qualified development evidence.
# cells: failure-{default,plain}-{off,on}  fault-{default,plain}  identity-{default,plain}-{off,on}
set -uo pipefail
CELL=$1; MODEL=$2; BIN=$3; ROOT=$4; CACHE_MB=${5:-64}; LOCK_FD=${6:-}
GATE_ARGS=()
[[ -n $LOCK_FD ]] && GATE_ARGS=(--external-lock "$LOCK_FD")
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/$CELL"; mkdir -p "$OUT"
case "$CELL" in
    failure-*) GATE=tools/kv-host-spill-failure-gate.sh ;;
    fault-*) GATE=tools/kv-host-contract-fault-gate.sh ;;
    identity-*) GATE=tools/kv-host-spill-identity-gate.sh ;;
    *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
ENVS=("MEMRA_HOSTGATE_CACHE_MB=$CACHE_MB")
[[ $CELL == *-plain* ]] && ENVS+=("MEMRA_SERVE_SPEC=0")
[[ $CELL == *-on ]] && ENVS+=("MEMRA_KV_HOST_CONTRACTS=1")
{
    echo "cell=$CELL gate=$GATE"
    echo "env=${ENVS[*]}"
    echo "lock=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock} owner=${LOCK_FD:+collector}${LOCK_FD:-internal-canonical}"
    echo "tree=$(git -C "$HERE" rev-parse HEAD)"
    echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "status=executed-not-qualified"
} > "$OUT/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.before.csv" 2>&1
rc=2
for attempt in $(seq 1 15); do
    date -u +%FT%TZ > "$OUT/started.txt"
    env "${ENVS[@]}" bash "$HERE/$GATE" "${GATE_ARGS[@]}" "$MODEL" "$BIN" "$OUT/ev" > "$OUT/gate.log" 2>&1
    rc=$?
    if [ -z "$LOCK_FD" ] && [ $rc -eq 2 ] && grep -q "REFUSED: canonical GPU lock busy" "$OUT/gate.log"; then
        echo "attempt $attempt: lock busy, waiting 120 s" >> "$OUT/lock-retries.txt"
        sleep 120
        continue
    fi
    break
done
date -u +%FT%TZ > "$OUT/finished.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.after.csv" 2>&1
echo "$rc" > "$OUT/gate.exit"
grep -E "GATE: " "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt"
echo "$CELL rc=$rc $(cat "$OUT/verdict.txt")"
exit "$rc"
