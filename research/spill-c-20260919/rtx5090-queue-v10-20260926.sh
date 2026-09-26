#!/usr/bin/env bash
# Lane C RTX 5090 queue v10 (2026-09-26): DAY72's cell gap15 on the RTX 5090, after queue v9 (its pid, by number)
# exits; the binary run-gen-i15 that v9 built in target/c-bins. Behind /tmp/memra-5090.lock with v9's idle wait; the
# runner inside the 1200% CPU cap; the reader on /usr/bin/python3; the profiler's reports and exports then move to
# profiles-hash-only/ with their SHA-256 (DAY72 section 1a). Never signals another process.
# usage: bash rtx5090-queue-v10-20260926.sh <v9 pid>
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
LOCK=/tmp/memra-5090.lock
BINS=$T/target/c-bins
ART=/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
V9=${1:?v9 pid}
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
log "queue v10 start: waits for v9 (pid $V9)"
while kill -0 "$V9" 2>/dev/null; do sleep 30; done
log "v9 exited"
[ -x "$BINS/run-gen-i15" ] || { log "missing run-gen-i15, stopping"; exit 1; }
R=$L/rtx5090-day72; name=gap15
mkdir -p "$R/builds"; printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.json -whitespace\n*.tsv -whitespace\n' > "$R/.gitattributes"
cp "$BINS/logs/build-i15.log" "$R/builds/"
[ -f "$R/$name.done" ] && { log "$name already done"; exit 0; }
idle 40 || { log "$name: rig never idle in 48 h"; exit 1; }
systemd-run --user --scope -q -p CPUQuota=1200% env PATH="$PATH" D40_POLL_S=5 D40_LOG_EVERY=24 D40_WAITS=4320 \
    D40_LOCK_TRIES=720 D40_MIN_AVAIL_GB=40 D40_RIG=rtx5090 D40_R="$R" D40_TREE=$T D40_CELL_SCRIPT="$L/day72-cell.sh" \
    D40_BINS="$BINS" D40_ART=$ART D40_LOCK=$LOCK bash "$L/day40-run-cell.sh" "$name" 7200
log "$name runner rc=$?"
out=$R/$name
for n in $(seq 30 -1 1); do [ -f "$R/$name-retry$n/CELL.jsonl" ] && { out=$R/$name-retry$n; break; }; done
/usr/bin/python3 "$T/tools/tier-battery.py" --rig rtx5090 --validate "$out" > "$R/$name-validate.log" 2>&1
log "$name validate rc=$?"
mkdir -p "$R/$name"
/usr/bin/python3 "$L/day72-read.py" "$R/$name" --rig rtx5090 > "$R/$name/reading.log" 2>&1
log "$name reader rc=$?"
P=$T/target/c-profiles-day72; mkdir -p "$P"
for f in "$out"/ev/*.nsys-rep "$out"/ev/*.sqlite; do [ -f "$f" ] && mv "$f" "$P/"; done
(cd "$P" && sha256sum -- * 2>/dev/null) > "$R/profiles.sha256"
touch "$R/$name.done"
log "queue v10 done"
