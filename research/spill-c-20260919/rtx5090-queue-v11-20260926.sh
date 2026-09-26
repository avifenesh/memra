#!/usr/bin/env bash
# Lane C RTX 5090 queue v11 (2026-09-26): new holds of the two 5090 cells that read inadmissible (DAY64 section 6,
# DAY72 section 2), the same cells, binaries (target/c-bins, built by queue v9) and readers, into new receipt dirs
# (rtx5090-day64-rerun1/, rtx5090-day72-rerun1/); the admissibility ceiling does not move. Behind the rig lock with
# v9's idle wait, each runner inside the 1200% CPU cap; never signals another process. gap15's profiler reports and
# exports then move to target/ with their SHA-256 in the cell's profiles.sha256.
# usage: bash rtx5090-queue-v11-20260926.sh   (log to target/c-queue/queue-v11.log)
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
LOCK=/tmp/memra-5090.lock
BWT=$T/target/c-build-wt
BINS=$T/target/c-bins
ART=/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
log() { echo "$(date -u +%FT%TZ) $*"; }
idle() { # $1 min MemAvailable GiB; waits up to 48 h, a line every 30 min
    local n=0
    while [ $n -lt 34560 ]; do
        if flock -n "$LOCK" true 2>/dev/null \
           && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
           && [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge $(( $1 * 1024 * 1024 )) ]; then
            return 0
        fi
        n=$((n + 1)); [ $((n % 360)) -eq 0 ] && log "waiting for an idle rig ($((n / 720)) h)"
        sleep 5
    done
    return 1
}
log "queue v11 start at $(git -C "$T" rev-parse --short HEAD)"
for b in run-gen-c60 run-gen-i13 run-gen-i14 run-gen-i15; do [ -x "$BINS/$b" ] || { log "missing $b, stopping"; exit 1; }; done
cell() { # $1 day  $2 cell  $3 min_avail_gb  $4 cell script  $5 reader command ({} = the cell dir)  [env words...]
    local day=$1 name=$2 gb=$3 script=$4 reader=$5; shift 5
    local R=$L/rtx5090-day$day-rerun1
    mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.json -whitespace\n*.tsv -whitespace\n' > "$R/.gitattributes"
    [ -f "$R/$name.done" ] && { log "$name already done"; return 0; }
    mkdir -p "$R/builds" && cp "$BINS"/logs/build-*.log "$R/builds/"
    idle "$gb" || { log "$name: rig never idle in 48 h"; return 1; }
    systemd-run --user --scope -q -p CPUQuota=1200% env PATH="$PATH" D40_POLL_S=5 D40_LOG_EVERY=24 D40_WAITS=4320 \
        D40_LOCK_TRIES=720 D40_MIN_AVAIL_GB="$gb" D40_RIG=rtx5090 D40_R="$R" D40_TREE=$T D40_CELL_SCRIPT="$script" \
        D40_BINS="$BINS" D40_ART=$ART D40_LOCK=$LOCK "$@" bash "$L/day40-run-cell.sh" "$name" 5400
    log "$name runner rc=$?"
    local out=$R/$name n
    for n in $(seq 30 -1 1); do [ -f "$R/$name-retry$n/CELL.jsonl" ] && { out=$R/$name-retry$n; break; }; done
    /usr/bin/python3 "$T/tools/tier-battery.py" --rig rtx5090 --validate "$out" > "$R/$name-validate.log" 2>&1
    log "$name validate rc=$?"
    mkdir -p "$R/$name"
    # The reader command takes the cell dir where it says {}, else after its words.
    local cmd=${reader//\{\}/$R/$name}
    [ "$cmd" = "$reader" ] && cmd="$reader $R/$name"
    /usr/bin/python3 $cmd --rig rtx5090 > "$R/$name/reading.log" 2>&1
    log "$name reader rc=$?"
    touch "$R/$name.done"
}
cell 64 i15 40 "$L/day64-cell.sh" "$L/day64-read.py {} --admissibility"
cell 72 gap15 40 "$L/day72-cell.sh" "$L/day72-read.py"
R=$L/rtx5090-day72-rerun1; P=$T/target/c-profiles-day72-rerun1; mkdir -p "$P"
for f in "$R"/gap15*/ev/*.nsys-rep "$R"/gap15*/ev/*.sqlite; do [ -f "$f" ] && mv "$f" "$P/"; done
(cd "$P" && sha256sum -- * 2>/dev/null) > "$R/profiles.sha256"
log "queue v11 done"
