#!/usr/bin/env bash
# day56-cell.sh <cell> <model.gguf> <drafter_export_dir> <server_bin> <out_root> [cache_mb]
# The day-24 cell driver shape (DAY56.md section 1): the host-tier identity gate's drafter arm on the 27B with the
# DFlash2 drafter (MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=<export dir> MEMRA_DSPARK_PREFIX_RESTORE=1), door OFF and
# door ON. The gate takes the canonical lock itself; a busy lock is retried 90 x 120 s and the holder is never
# inspected or signalled. Records nvidia-smi compute apps and driver free before and after, the binary digest, the
# tree SHA, the drafter's two files' SHA-256, the verdict line and the exit code. Executed-not-qualified evidence.
# cells: identity-dspark-{off,on}
set -uo pipefail
CELL=$1; MODEL27=$2; DRAFT=$3; BIN=$4; ROOT=$5; CACHE_MB=${6:-256}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/$CELL"; mkdir -p "$OUT"
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
export MEMRA_GPU_LOCK=$LOCK
MODEL=$MODEL27
case "$CELL" in
    identity-dspark-*) GATE=tools/kv-host-spill-identity-gate.sh ;;
    *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
# DAY56 section 2c: the gate's drafter arm re-sends P_A (a whole-cover restore), so the strict-prefix switch of
# attempts 2 and 3 (MEMRA_DSPARK_PARTIAL_RESTORE=1) is not set.
ENVS=("MEMRA_HOSTGATE_CACHE_MB=$CACHE_MB" MEMRA_DSPARK_SPEC=1 "MEMRA_DSPARK_DRAFT=$DRAFT" MEMRA_DSPARK_PREFIX_RESTORE=1)
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
    echo "drafter_config_sha256=$(sha256sum "$DRAFT/config.json" | cut -d' ' -f1)"
    echo "drafter_model_sha256=$(sha256sum "$DRAFT/model.safetensors" | cut -d' ' -f1)"
    if [[ -n $MODEL ]]; then echo "model=$(basename "$MODEL")"; else echo "model=none (unit cell: the test binary opens its own device context)"; fi
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "status=executed-not-qualified"
} > "$OUT/CELL.txt"
snap before
rc=2
for attempt in $(seq 1 90); do
    date -u +%FT%TZ > "$OUT/started.txt"
    case "$CELL" in
        unit-*)
            ( cd "$HERE" || exit 3
              if ! flock -n 9; then echo "REFUSED: canonical GPU lock busy ($LOCK)"; exit 2; fi
              python3 tools/tier-lock-proof.py --fd 9 --lock "$LOCK" --owner day53-cell > "$OUT/LOCK.json" 2>&1
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
    unit-*) grep -E "^test result" "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt" ;;
    *) grep -E "GATE: " "$OUT/gate.log" | tail -1 > "$OUT/verdict.txt" ;;
esac
echo "$CELL rc=$rc $(cat "$OUT/verdict.txt")"
exit "$rc"
