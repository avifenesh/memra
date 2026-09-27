#!/usr/bin/env bash
# WP-C RTX 5090 queue v6: DAY59's four cells, DAY60's gap and DAY61's i11 on the local card, behind queue v5 (it waits
# for v5 to exit and for the local c60, i11 and i12 builds), each cell through the day-40 runner under the 1200% cap
# and /tmp/memra-5090.lock, the registered reader after each. The idle poll (lock free, no compute app, host memory)
# holds every cell until the card is healthy. Nothing seen is touched.
set -uo pipefail
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
LOCK=/tmp/memra-5090.lock
V5=${V5_PID:-3557213}
log() { echo "$(date -u +%FT%TZ) $*"; }
idle() { # $1 min MemAvailable GiB; bounded 6 h
    for _ in $(seq 1 4320); do
        if flock -n "$LOCK" true 2>/dev/null \
           && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
           && [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge $(( $1 * 1024 * 1024 )) ]; then
            return 0
        fi
        sleep 5
    done
    return 1
}
log "queue v6 start: waits for v5 (pid $V5) and the local builds"
while kill -0 "$V5" 2>/dev/null; do sleep 30; done
log "v5 exited"
until grep -q "all builds done" /tmp/c61-build/build-local.log 2>/dev/null; do
    grep -q "failed" /tmp/c61-build/build-local.log 2>/dev/null && { log "local build failed, stopping"; exit 1; }
    sleep 30
done
log "builds done"
cell() { # $1 day  $2 cell  $3 min_avail_gb  $4 cell script  $5 reader command (after the cell dir)
    local R=$L/rtx5090-day$1
    mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.json -whitespace\n' > "$R/.gitattributes"
    [ -f "$R/$2.done" ] && { log "$2 already done"; return 0; }
    idle "$3" || { log "$2: rig never idle in 6 h"; return 1; }
    systemd-run --user --scope -q -p CPUQuota=1200% env D40_POLL_S=5 D40_LOG_EVERY=24 D40_WAITS=4320 D40_LOCK_TRIES=720 \
        D40_MIN_AVAIL_GB="$3" D40_RIG=rtx5090 D40_R="$R" D40_TREE=$T D40_CELL_SCRIPT="$4" \
        D40_BINS=/tmp/c40-bins D40_ART=/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
        D40_LOCK=$LOCK bash "$L/day40-run-cell.sh" "$2" 5400
    log "$2 runner rc=$?"
    local out=$R/$2 n
    for n in $(seq 30 -1 1); do [ -f "$R/$2-retry$n/CELL.jsonl" ] && { out=$R/$2-retry$n; break; }; done
    python3 "$T/tools/tier-battery.py" --rig rtx5090 --validate "$out" > "$R/$2-validate.log" 2>&1
    log "$2 validate rc=$?"
    mkdir -p "$R/$2"
    # shellcheck disable=SC2086
    python3 $5 "$R/$2" --rig rtx5090 > "$R/$2/reading.log" 2>&1
    log "$2 reader rc=$?"
    touch "$R/$2.done"
}
mkdir -p "$L/rtx5090-day61/builds" && cp /tmp/c61-build/build-*.log "$L/rtx5090-day61/builds/"
cell 59 pfgates 40 "$L/day59-cell.sh" "$L/day59-pf.py gates"
cell 59 pfserve 40 "$L/day59-cell.sh" "$L/day59-pf.py serve"
cell 59 pftime 40 "$L/day59-cell.sh" "$L/day59-pf.py time"
cell 59 pfnaked 40 "$L/day59-cell.sh" "$L/day59-pf.py time"
cell 60 gap 40 "$L/day60-cell.sh" "$L/day60-gap.py"
cell 61 i11 40 "$L/day61-cell.sh" "$L/day61-read.py"
log "queue v6 done"
