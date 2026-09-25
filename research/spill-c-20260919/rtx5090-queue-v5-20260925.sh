#!/usr/bin/env bash
# WP-C RTX 5090 queue v5 (replaces v3 and v4): DAY51's hashlock, spec and decide-b (section 1c) on the final tree,
# then the day-56 drafter cells in their attempt-4 shape (DAY56 section 2c). The card required a reset when this
# started; the idle poll (lock free, no compute app) holds every cell until the card is healthy. Nothing seen is
# touched.
set -uo pipefail
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
LOCK=/tmp/memra-5090.lock
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
log "queue v5 start"
moe() { # $1 day  $2 cell  $3 min_avail_gb  $4 cell script
    local R=$L/rtx5090-day$1
    mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n' > "$R/.gitattributes"
    idle "$3" || { log "$2: rig never idle in 6 h"; return 1; }
    systemd-run --user --scope -q -p CPUQuota=1200% env D40_POLL_S=5 D40_LOG_EVERY=24 D40_WAITS=4320 D40_LOCK_TRIES=720 \
        D40_MIN_AVAIL_GB="$3" D40_RIG=rtx5090 D40_R="$R" D40_TREE=$T D40_CELL_SCRIPT="$4" \
        D40_BINS=/tmp/c40-bins D40_ART=/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
        D40_LOCK=$LOCK D40_ART_OTHER=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf \
        D51_MOE_ENV= bash "$L/day40-run-cell.sh" "$2" 5400
    log "$2 runner rc=$?"
}
moe 51 hashlock 20 "$L/day51-cell.sh"
moe 51 spec 40 "$L/day51-cell.sh"
moe 51 decide-b 40 "$L/day51-cell.sh"
R=$L/rtx5090-day56; mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.csv -whitespace\n' > "$R/.gitattributes"
M27=/home/avifenesh/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for c in identity-dspark-off identity-dspark-on; do
    idle 30 || { log "$c: rig never idle"; continue; }
    systemd-run --user --scope -q -p CPUQuota=1200% bash "$L/day56-cell.sh" "$c" "$M27" /data/ai-ml/models/q38-dflash2 /tmp/c53-bins/memra-server-c5 "$R" 256
    log "c5 $c rc=$?"
done
python3 "$L/day56-reading.py" "$R" --rig rtx5090 > "$R/reading.log" 2>&1
log "queue v5 done (c5 reader rc=$?)"
