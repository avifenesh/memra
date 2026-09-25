#!/usr/bin/env bash
# WP-C RTX 5090 queue v4: after v3 exits (DAY51's cells), the day-56 drafter cells in their attempt-4 shape
# (DAY56 section 2c). The same idle poll and cap as v3; nothing seen is touched.
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
log "queue v4 start: waits for v3 to exit"
while pgrep -f "c53-smoke/queue3.sh" > /dev/null; do sleep 30; done
R=$L/rtx5090-day56; mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.csv -whitespace\n' > "$R/.gitattributes"
M27=/home/avifenesh/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for c in identity-dspark-off identity-dspark-on; do
    idle 30 || { log "$c: rig never idle"; continue; }
    systemd-run --user --scope -q -p CPUQuota=1200% bash "$L/day56-cell.sh" "$c" "$M27" /data/ai-ml/models/q38-dflash2 /tmp/c53-bins/memra-server-c5 "$R" 256
    log "c5 $c rc=$?"
done
python3 "$L/day56-reading.py" "$R" --rig rtx5090 > "$R/reading.log" 2>&1
log "queue v4 done (c5 reader rc=$?)"
