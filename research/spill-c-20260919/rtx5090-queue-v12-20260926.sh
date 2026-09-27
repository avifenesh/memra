#!/usr/bin/env bash
# Lane C RTX 5090 queue v12 (2026-09-26): DAY75's cell i16 on the RTX 5090. First run-gen-i16 built from its commit
# by c-local-build.sh in the build worktree under target/ (inside the CPU cap, into target/c-bins, beside v9's
# run-gen-i15). Behind /tmp/memra-5090.lock with v9's idle wait; the
# runner inside the 1200% CPU cap; the reader on /usr/bin/python3; the profiler's reports and exports then move to
# target/ with their SHA-256 in the cell's profiles.sha256 (DAY72 section 1a). Never signals another process.
# usage: bash rtx5090-queue-v12-20260926.sh
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
LOCK=/tmp/memra-5090.lock
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
log "queue v12 start at $(git -C "$T" rev-parse --short HEAD)"
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G bash "$L/c-local-build.sh" "$T/target/c-build-wt" "$BINS" \
    i16=eeacfaf50:memra-engine/run-gen
rc=$?; log "build rc=$rc"; [ "$rc" -eq 0 ] || exit 1
for b in run-gen-i15 run-gen-i16; do [ -x "$BINS/$b" ] || { log "missing $b, stopping"; exit 1; }; done
R=$L/rtx5090-day75; name=i16
mkdir -p "$R/builds"; printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.json -whitespace\n*.tsv -whitespace\n' > "$R/.gitattributes"
cp "$BINS/logs/build-i15.log" "$BINS/logs/build-i16.log" "$R/builds/"
[ -f "$R/$name.done" ] && { log "$name already done"; exit 0; }
idle 40 || { log "$name: rig never idle in 48 h"; exit 1; }
systemd-run --user --scope -q -p CPUQuota=1200% env PATH="$PATH" D40_POLL_S=5 D40_LOG_EVERY=24 D40_WAITS=4320 \
    D40_LOCK_TRIES=720 D40_MIN_AVAIL_GB=40 D40_RIG=rtx5090 D40_R="$R" D40_TREE=$T D40_CELL_SCRIPT="$L/day75-cell.sh" \
    D40_BINS="$BINS" D40_ART=$ART D40_LOCK=$LOCK bash "$L/day40-run-cell.sh" "$name" 7200
log "$name runner rc=$?"
out=$R/$name
for n in $(seq 30 -1 1); do [ -f "$R/$name-retry$n/CELL.jsonl" ] && { out=$R/$name-retry$n; break; }; done
/usr/bin/python3 "$T/tools/tier-battery.py" --rig rtx5090 --validate "$out" > "$R/$name-validate.log" 2>&1
log "$name validate rc=$?"
mkdir -p "$R/$name"
/usr/bin/python3 "$L/day75-read.py" "$R/$name" --rig rtx5090 > "$R/$name/reading.log" 2>&1
log "$name reader rc=$?"
P=$T/target/c-profiles-day75; mkdir -p "$P"
for f in "$out"/ev/*.nsys-rep "$out"/ev/*.sqlite; do [ -f "$f" ] && mv "$f" "$P/"; done
(cd "$P" && sha256sum -- * 2>/dev/null) > "$R/profiles.sha256"
touch "$R/$name.done"
log "queue v12 done"
