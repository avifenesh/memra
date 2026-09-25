#!/usr/bin/env bash
# WP-C RTX 5090 queue (2026-09-24, replaces chains 2 to 6): one cell after another, in the order the ledger needs
# them (C1's rungs and I10 first, then C6's gates, C8's cell, C5's cells). Before each cell a bounded poll every 5 s
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
moe() { # $1 day  $2 cell  $3 min_avail_gb
    local R=$L/rtx5090-day$1
    mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n' > "$R/.gitattributes"
    idle "$3" || { log "$2: rig never idle in 6 h"; return 1; }
    systemd-run --user --scope -q -p CPUQuota=1200% env D40_POLL_S=5 D40_LOG_EVERY=24 D40_WAITS=4320 D40_LOCK_TRIES=720 \
        D40_MIN_AVAIL_GB="$3" D40_RIG=rtx5090 D40_R="$R" D40_TREE=$T D40_CELL_SCRIPT="$L/day$1-cell.sh" \
        D40_BINS=/tmp/c40-bins D40_ART=$ART D40_LOCK=$LOCK bash "$L/day40-run-cell.sh" "$2" 5400
    log "$2 runner rc=$?"
}
log "queue start"
moe 43 resid 46
moe 44 mapped 36
moe 45 fill 40
moe 46 nodrain 40
moe 47 pinned 40
moe 48 small 40
moe 49 install 40
moe 50 prefetch 40
moe 57 fillwait 40
# C6 (DAY53): the verify digest v3 gates on the 9B.
R=$L/rtx5090-day53; mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.csv -whitespace\n' > "$R/.gitattributes"
M9=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
for c in unit-server failure-default-off failure-plain-off failure-default-on identity-default-off identity-default-on; do
    idle 20 || { log "$c: rig never idle"; continue; }
    systemd-run --user --scope -q -p CPUQuota=1200% bash "$L/day53-cell.sh" "$c" "$M9" /tmp/c53-bins/memra-server-v3 "$R" 64
    log "c6 $c rc=$?"
done
# C8 (DAY55): the always-admitted prime arm.
R=$L/rtx5090-day55; mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.csv -whitespace\n' > "$R/.gitattributes"
idle 20 && systemd-run --user --scope -q -p CPUQuota=1200% bash "$L/day55-local-run.sh" "$R" /tmp/c53-bins/memra-server-v3
log "c8 rc=$?"
# C5 (DAY56): the identity gate's drafter arm on the 27B.
R=$L/rtx5090-day56; mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.csv -whitespace\n' > "$R/.gitattributes"
M27=/home/avifenesh/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for c in identity-dspark-off identity-dspark-on; do
    idle 30 || { log "$c: rig never idle"; continue; }
    systemd-run --user --scope -q -p CPUQuota=1200% bash "$L/day56-cell.sh" "$c" "$M27" /data/ai-ml/models/q38-dflash2 /tmp/c53-bins/memra-server-c5 "$R" 256
    log "c5 $c rc=$?"
done
python3 "$L/day56-reading.py" "$R" --rig rtx5090 > "$R/reading.log" 2>&1
log "queue done (c5 reader rc=$?)"
