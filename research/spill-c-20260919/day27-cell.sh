#!/usr/bin/env bash
# day27-cell.sh <hit-off|hit-on> <model.gguf> <server_bin> <out_root>
# The hit gate (`tools/spec-on-cache-hit-gate.sh qwen`) under one door arm, on the tree whose gate arms the host
# tier in its ON arm (C day 27): `hit-off` runs the gate with MEMRA_KV_HOST_CONTRACTS unset, `hit-on` with
# MEMRA_KV_HOST_CONTRACTS=1 (the gate itself then exports MEMRA_KV_HOST_MB=8192 to both boots and asserts the
# door engaged). The gate takes the canonical lock itself (`flock -w 300` per boot on MEMRA_GPU_LOCK; it has no
# `--external-lock`, so it cannot run under the collector's hold: lane A day 21 and C day 26 ran it the same
# way). Before the gate this driver waits, bounded (15 x 120 s), for the canonical lock to be free (a `flock -n`
# probe that holds nothing) and never inspects or signals a holder. Records nvidia-smi compute apps and driver
# free before and after, the binary digest, the tree SHA, the verdict line and the exit code, the door-line
# census (`door-lines.txt`: every arming, door, route, refusal and latch line under the cell) and the restore
# census (`restore-lines.txt`, the day-26 shape). Every cell is executed-not-qualified development evidence.
set -uo pipefail
CELL=$1; MODEL=$2; BIN=$3; ROOT=$4
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/$CELL"; mkdir -p "$OUT"
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
export MEMRA_GPU_LOCK=$LOCK
GATE=tools/spec-on-cache-hit-gate.sh
ENVS=()
case "$CELL" in
    hit-off) ;;
    hit-on) ENVS+=("MEMRA_KV_HOST_CONTRACTS=1") ;;
    *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
snap() { # label
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.$1.csv" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv > "$OUT/card.$1.csv" 2>&1
}
{
    echo "cell=$CELL gate=$GATE"
    echo "env=${ENVS[*]:-}"
    echo "lock=$LOCK owner=gate-internal-canonical (flock -w 300 per boot inside the gate)"
    echo "tree=$(git -C "$HERE" rev-parse HEAD)"
    echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "status=executed-not-qualified"
} > "$OUT/CELL.txt"
for attempt in $(seq 1 15); do
    if flock -n "$LOCK" true 2>/dev/null; then break; fi
    echo "$(date -u +%FT%TZ) attempt $attempt: canonical lock $LOCK busy, waiting 120 s" >> "$OUT/lock-retries.txt"
    sleep 120
done
snap before
date -u +%FT%TZ > "$OUT/started.txt"
env "${ENVS[@]}" bash "$HERE/$GATE" qwen "$MODEL" "$BIN" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
date -u +%FT%TZ > "$OUT/finished.txt"
snap after
echo "$rc" > "$OUT/gate.exit"
grep -E "GATE: " "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt"
grep -rHE "\[prefix-host\] on: budget|contracts door ON|\[kv-host-contracts\]|(capture|restore|demote|promote) (submitted|published|landed) off the tick|refused \(contracts door\)|restore refused|restore dropped|capture dropped|TIER DISABLED|OFF-TICK DISABLED" "$OUT/ev" 2>/dev/null | sort > "$OUT/door-lines.txt"
grep -rHE "restore submitted off the tick|restore landed off the tick|restore refused|restore dropped|RESTORE OFF-TICK DISABLED" "$OUT/ev" 2>/dev/null | sort > "$OUT/restore-lines.txt"
echo "door_lines=$(wc -l < "$OUT/door-lines.txt") restore_route_lines=$(wc -l < "$OUT/restore-lines.txt")" >> "$OUT/CELL.txt"
echo "$CELL rc=$rc $(cat "$OUT/verdict.txt") door_lines=$(wc -l < "$OUT/door-lines.txt") restore_route_lines=$(wc -l < "$OUT/restore-lines.txt")"
exit "$rc"
