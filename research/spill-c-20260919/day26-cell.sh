#!/usr/bin/env bash
# day26-cell.sh <cell> <model9b.gguf> <model27b.gguf> <server_bin> <out_root> [cache_mb]
# The day-24 cell driver on the Move 2 slice-2 tree (A `ff3f0f6f5` through integ37): the same sixteen cells,
# with the engine unit cell widened to both D2D classes (`d2d_capture_*` and the new `d2d_restore_*`), and a
# census of the restore route per cell (every `restore submitted off the tick`, `restore landed off the tick`,
# `restore refused`, `restore dropped` and `RESTORE OFF-TICK DISABLED` line under the cell, into `restore-lines.txt`).
# The hit gate (9B, `spec-on-cache-hit-gate.sh qwen`) and the twin gate (27B, `prefix-newest-turn-fits-gate.py`,
# `REFUSED: V3 premise` rc=2 is a refusal to report and not a FAIL) take the canonical lock themselves; the unit
# cells run under this driver's own `flock -n` on the canonical lock. Every gate that takes the lock
# itself is left to do so; a busy lock is retried 15 x 120 s and the holder is never inspected or signalled.
# Records nvidia-smi compute apps and driver free before and after, the binary digest, the tree SHA, the
# verdict line and the exit code. Every cell is executed-not-qualified development evidence.
# cells: failure-{default,plain}-{off,on}  fault-{default,plain}  identity-{default,plain}-{off,on}
#        hit-{off,on}  twin27-{off,on}  unit-server  unit-engine
set -uo pipefail
CELL=$1; MODEL9=$2; MODEL27=$3; BIN=$4; ROOT=$5; CACHE_MB=${6:-64}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/$CELL"; mkdir -p "$OUT"
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
export MEMRA_GPU_LOCK=$LOCK
MODEL=$MODEL9
case "$CELL" in
    failure-*) GATE=tools/kv-host-spill-failure-gate.sh ;;
    fault-*) GATE=tools/kv-host-contract-fault-gate.sh ;;
    identity-*) GATE=tools/kv-host-spill-identity-gate.sh ;;
    hit-*) GATE=tools/spec-on-cache-hit-gate.sh ;;
    twin27-*) GATE=tools/prefix-newest-turn-fits-gate.py; MODEL=$MODEL27 ;;
    unit-server) GATE="cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_"; MODEL= ;;
    unit-engine) GATE="cargo test -p memra-engine --lib -- --ignored --test-threads=1 d2d_"; MODEL= ;;
    *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
ENVS=()
[[ $CELL == failure-* || $CELL == fault-* || $CELL == identity-* ]] && ENVS+=("MEMRA_HOSTGATE_CACHE_MB=$CACHE_MB")
[[ $CELL == *-plain* ]] && ENVS+=("MEMRA_SERVE_SPEC=0")
[[ $CELL == *-on ]] && ENVS+=("MEMRA_KV_HOST_CONTRACTS=1")
snap() { # label
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.$1.csv" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv > "$OUT/card.$1.csv" 2>&1
}
{
    echo "cell=$CELL gate=$GATE"
    echo "env=${ENVS[*]:-}"
    echo "lock=$LOCK owner=$([[ $CELL == unit-* ]] && echo driver-flock || echo gate-internal-canonical)"
    echo "tree=$(git -C "$HERE" rev-parse HEAD)"
    echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    if [[ -n $MODEL ]]; then echo "model=$(basename "$MODEL")"; else echo "model=none (unit cell: the test binary opens its own device context)"; fi
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "status=executed-not-qualified"
} > "$OUT/CELL.txt"
snap before
rc=2
for attempt in $(seq 1 15); do
    date -u +%FT%TZ > "$OUT/started.txt"
    case "$CELL" in
        twin27-*)
            rm -rf "$OUT/ev"
            env "${ENVS[@]}" python3 "$HERE/$GATE" --model "$MODEL" --bin "$BIN" --out "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$? ;;
        hit-*)
            env "${ENVS[@]}" bash "$HERE/$GATE" qwen "$MODEL" "$BIN" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$? ;;
        unit-*)
            ( cd "$HERE" || exit 3
              if ! flock -n 9; then echo "REFUSED: canonical GPU lock busy ($LOCK)"; exit 2; fi
              python3 tools/tier-lock-proof.py --fd 9 --lock "$LOCK" --owner day26-cell > "$OUT/LOCK.json" 2>&1
              # shellcheck disable=SC2086
              CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G --quiet $GATE
            ) 9>>"$LOCK" > "$OUT/gate.log" 2>&1; rc=$? ;;
        *)
            env "${ENVS[@]}" bash "$HERE/$GATE" "$MODEL" "$BIN" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$? ;;
    esac
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
case "$CELL" in
    twin27-*) grep -hE "PREFIX-NEWEST-TURN-FITS|REFUSED" "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt" ;;
    unit-*) grep -E "^test result" "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt" ;;
    *) grep -E "GATE: " "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt" ;;
esac
# The restore route's census for this cell (WP-A day 21 lines), verbatim, with the file each came from.
grep -rHE "restore submitted off the tick|restore landed off the tick|restore refused|restore dropped|RESTORE OFF-TICK DISABLED" "$OUT/ev" 2>/dev/null | sort > "$OUT/restore-lines.txt"
echo "restore_route_lines=$(wc -l < "$OUT/restore-lines.txt")" >> "$OUT/CELL.txt"
echo "$CELL rc=$rc $(cat "$OUT/verdict.txt") restore_route_lines=$(wc -l < "$OUT/restore-lines.txt")"
exit "$rc"
