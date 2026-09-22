#!/usr/bin/env bash
# day31-cell.sh <cell> <model9b.gguf> <server_bin> <out_root> [cache_mb]
# The whole-budget failure arm on the local RTX 5090 Laptop GPU (HOSTPREFIX-DOOR.md section D item 9; the
# packet's "not run" row): tools/kv-host-spill-failure-gate.sh with MEMRA_KV_HOST_TENANT_PCT=100 (the share cap
# disarmed, so the pool-full cell's refusal is the insert-path line `skip demote: entry X MB > host budget B MB`
# after the copy), default (spec) and plain, door OFF and ON, in the shape of the day-23 target-card cells
# (pro-single-day23-gates/cells/failure-default-pct100-{off,on}) through the day-26 local cell driver. The gate
# takes the canonical lock itself; a busy lock is retried 15 x 120 s and the holder is never inspected or
# signalled. Records nvidia-smi compute apps and the card before and after, the binary digest, the tree SHA, the
# verdict line, the exit code, and the census of the pool-full cell's demote lines (`skip demote`, `demote
# evaporated`, `demote submitted`, `D2H receipt`, `demote published`). Every cell is executed-not-qualified
# development evidence.
# cells: failure-{default,plain}-pct100-{off,on}
set -uo pipefail
CELL=$1; MODEL=$2; BIN=$3; ROOT=$4; CACHE_MB=${5:-64}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/$CELL"; mkdir -p "$OUT"
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
export MEMRA_GPU_LOCK=$LOCK
case "$CELL" in
    failure-*-pct100-*) GATE=tools/kv-host-spill-failure-gate.sh ;;
    *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
ENVS=("MEMRA_HOSTGATE_CACHE_MB=$CACHE_MB" "MEMRA_KV_HOST_TENANT_PCT=100")
[[ $CELL == *-plain-* ]] && ENVS+=("MEMRA_SERVE_SPEC=0")
[[ $CELL == *-on ]] && ENVS+=("MEMRA_KV_HOST_CONTRACTS=1")
snap() { # label
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.$1.csv" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv > "$OUT/card.$1.csv" 2>&1
}
{
    echo "cell=$CELL gate=$GATE"
    echo "env=${ENVS[*]}"
    echo "lock=$LOCK owner=gate-internal-canonical"
    echo "tree=$(git -C "$HERE" rev-parse HEAD)"
    echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "status=executed-not-qualified"
} > "$OUT/CELL.txt"
snap before
rc=2
for attempt in $(seq 1 15); do
    date -u +%FT%TZ > "$OUT/started.txt"
    env "${ENVS[@]}" bash "$HERE/$GATE" "$MODEL" "$BIN" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
    if [ $rc -eq 2 ] && grep -q "REFUSED: canonical GPU lock busy" "$OUT/gate.log"; then
        echo "attempt $attempt: lock busy, waiting 120 s" >> "$OUT/lock-retries.txt"
        sleep 120
        continue
    fi
    break
done
date -u +%FT%TZ > "$OUT/finished.txt"
snap after
echo "$rc" > "$OUT/gate.exit"
grep -E "GATE: " "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt"
grep -hE "skip demote|demote evaporated|demote submitted off the tick|D2H receipt|demote published off the tick|demote refused|TIER DISABLED" "$OUT/ev/poolfull-server.log" > "$OUT/poolfull-demote-lines.txt" 2>/dev/null || true
echo "poolfull_demote_lines=$(wc -l < "$OUT/poolfull-demote-lines.txt")" >> "$OUT/CELL.txt"
echo "$CELL rc=$rc $(cat "$OUT/verdict.txt") poolfull_demote_lines=$(wc -l < "$OUT/poolfull-demote-lines.txt")"
exit "$rc"
