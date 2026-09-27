#!/usr/bin/env bash
# Lane C RTX 5090 queue v9 (2026-09-26): the one queue that replaces v5 to v8, which did not survive the rig's reboot
# (it wiped /tmp/c40-bins, /tmp/c53-bins and /tmp/c61-build). First the binaries, rebuilt from their exact commits by
# c-local-build.sh in a detached build worktree under the lane's target/ (inside the 1200% CPU cap), into
# target/c-bins; then the unrun cells in v5 to v8's order, each behind the rig lock /tmp/memra-5090.lock with an idle
# check (no compute app, enough MemAvailable), each runner inside the CPU cap. The card is shared with lanes A and B
# under the same lock; this queue never signals another process. Readers run with /usr/bin/python3.
# usage: bash rtx5090-queue-v9-20260926.sh   (log to target/c-queue/queue.log)
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
log "queue v9 start at $(git -C "$T" rev-parse --short HEAD)"
[ -d "$BWT" ] || git -C "$T" worktree add -q --detach "$BWT" HEAD || { log "build worktree failed"; exit 1; }
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G bash "$L/c-local-build.sh" "$BWT" "$BINS" \
    final=62e848b1f:memra-engine/run-gen,memra-engine/run-spec \
    c5=1b130f1ef:memra-server/memra-server \
    c60=da649107c:memra-engine/run-gen,memra-engine/run-spec,memra-server/memra-server \
    i11=a068ee37d:memra-engine/run-gen i12=117302725:memra-engine/run-gen i13=c9379c051:memra-engine/run-gen \
    i14=83f03d9b7:memra-engine/run-gen i15=2243b1fe2:memra-engine/run-gen
rc=$?
log "builds rc=$rc"
[ "$rc" -eq 0 ] || exit 1
cell() { # $1 day  $2 cell  $3 min_avail_gb  $4 cell script  $5 reader command ({} = the cell dir)  [env words...]
    local day=$1 name=$2 gb=$3 script=$4 reader=$5; shift 5
    local R=$L/rtx5090-day$day
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
O27=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
# v5: DAY51's G1, G2 and decide-b; then DAY56's C5 identity pair.
cell 51 hashlock 20 "$L/day51-cell.sh" "$L/day51-decide.py hashlock" D40_ART_OTHER=$O27 D51_MOE_ENV=
cell 51 spec 40 "$L/day51-cell.sh" "$L/day51-decide.py spec" D40_ART_OTHER=$O27 D51_MOE_ENV=
cell 51 decide-b 40 "$L/day51-cell.sh" "$L/day51-decide.py decide-b" D40_ART_OTHER=$O27 D51_MOE_ENV=
R=$L/rtx5090-day56; mkdir -p "$R"; printf '*.log -whitespace\n*.txt -whitespace\n*.csv -whitespace\n' > "$R/.gitattributes"
M27=/home/avifenesh/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
if [ ! -f "$R/c5.done" ]; then
    for c in identity-dspark-off identity-dspark-on; do
        idle 30 || { log "$c: rig never idle"; continue; }
        systemd-run --user --scope -q -p CPUQuota=1200% env PATH="$PATH" bash "$L/day56-cell.sh" "$c" "$M27" \
            /data/ai-ml/models/q38-dflash2 "$BINS/memra-server-c5" "$R" 256
        log "c5 $c rc=$?"
    done
    /usr/bin/python3 "$L/day56-reading.py" "$R" --rig rtx5090 > "$R/reading.log" 2>&1
    log "c5 reader rc=$?"; touch "$R/c5.done"
fi
# v6: DAY59's four prefetch cells, DAY60's gap, DAY61's i11.
cell 59 pfgates 40 "$L/day59-cell.sh" "$L/day59-pf.py gates"
cell 59 pfserve 40 "$L/day59-cell.sh" "$L/day59-pf.py serve"
cell 59 pftime 40 "$L/day59-cell.sh" "$L/day59-pf.py time"
cell 59 pfnaked 40 "$L/day59-cell.sh" "$L/day59-pf.py time"
cell 60 gap 40 "$L/day60-cell.sh" "$L/day60-gap.py"
cell 61 i11 40 "$L/day61-cell.sh" "$L/day61-read.py"
# v7 and v8: DAY63's i13 and DAY64's i15 (read with the admissibility clause, DAY64 section 5).
cell 63 i13 40 "$L/day63-cell.sh" "$L/day63-read.py"
cell 64 i15 40 "$L/day64-cell.sh" "$L/day64-read.py {} --admissibility"
log "queue v9 done"
