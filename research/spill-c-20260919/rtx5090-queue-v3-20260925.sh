#!/usr/bin/env bash
# WP-C RTX 5090 queue v3 (after v2: day 58 smallfix, then DAY51); from v2 (2026-09-24, replaces v1 after day 43's cell): one cell after another, in the order the
# ledger needs them (C1's rungs from day 44, I10, day 43's fix check, then C6's gates, C8's cell, C5's cells, then
# DAY51's three cells once the final tree is named: they wait for FINAL.ready, which the lane writes after it copies
# run-gen-final and run-spec-final into the binary dir). Before each cell a bounded poll every 5 s
# for an idle rig (the lock free, no compute app, the cell's MemAvailable), so the cell can start in the gap between
# another lane's boots; the runners keep their own bounded waits and lock retries and never touch a holder. Every
# runner inside a user scope capped at 1200% CPU.
set -uo pipefail
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
LOCK=/tmp/memra-5090.lock
ART=/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
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
moe() { # $1 day  $2 cell  $3 min_avail_gb  [$4 cell script, default day$1-cell.sh]
    local R=$L/rtx5090-day$1 S=${4:-$L/day$1-cell.sh}
    mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n' > "$R/.gitattributes"
    idle "$3" || { log "$2: rig never idle in 6 h"; return 1; }
    systemd-run --user --scope -q -p CPUQuota=1200% env D40_POLL_S=5 D40_LOG_EVERY=24 D40_WAITS=4320 D40_LOCK_TRIES=720 \
        D40_MIN_AVAIL_GB="$3" D40_RIG=rtx5090 D40_R="$R" D40_TREE=$T D40_CELL_SCRIPT="$S" \
        D40_BINS=/tmp/c40-bins D40_ART=$ART D40_LOCK=$LOCK bash "$L/day40-run-cell.sh" "$2" 5400
    log "$2 runner rc=$?"
}
log "queue v3 start: waits for v2 to reach its FINAL.ready wait, stops it there, runs day 58's smallfix, then DAY51"
until grep -q "c5 reader rc=" /tmp/c53-smoke/queue.log; do sleep 30; done
kill -TERM 2742907 2>/dev/null && log "queue v2 (pid 2742907) stopped by its owner in its FINAL.ready wait"
moe 58 smallfix 40
# DAY51 on the final tree: wait (bounded 24 h) for the lane to name it.
for _ in $(seq 1 17280); do [ -e /tmp/c53-smoke/FINAL.ready ] && break; sleep 5; done
if [ -e /tmp/c53-smoke/FINAL.ready ]; then
    export D40_ART_OTHER=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf D51_MOE_ENV=""
    moe 51 hashlock 20
    moe 51 spec 40
    moe 51 decide 40
else
    log "FINAL.ready never appeared: DAY51 cells not run"
fi
log "queue done"
