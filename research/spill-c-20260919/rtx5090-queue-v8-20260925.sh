#!/usr/bin/env bash
# WP-C RTX 5090 queue v8: DAY64's cell i15 on the local card, behind queue v7 (it waits
# for v7 to exit; the local run-gen-c60, -i13, -i14 and -i15 are built), the cell through the day-40 runner under the 1200% cap
# and /tmp/memra-5090.lock, the registered reader after each. The idle poll (lock free, no compute app, host memory)
# holds every cell until the card is healthy. Nothing seen is touched.
set -uo pipefail
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
LOCK=/tmp/memra-5090.lock
V7=${V7_PID:-45460}
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
log "queue v8 start: waits for v7 (pid $V7)"
while kill -0 "$V7" 2>/dev/null; do sleep 30; done
log "v7 exited"
for b in run-gen-c60 run-gen-i13 run-gen-i14 run-gen-i15; do [ -x "/tmp/c40-bins/$b" ] || { log "missing $b, stopping"; exit 1; }; done
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
mkdir -p "$L/rtx5090-day64/builds" && cp /tmp/c61-build/build-*.log "$L/rtx5090-day64/builds/"
cell 64 i15 40 "$L/day64-cell.sh" "$L/day64-read.py"
log "queue v8 done"
