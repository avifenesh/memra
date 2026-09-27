#!/usr/bin/env bash
# Lane C RTX 5090 queue v17 (2026-09-26): DAY83's cell `split20` (research/spill-c-20260919/DAY83.md section 1,
# registered before this script). Arms i15s (run-gen-i15 with --moe-dispatch-clock --expert-bank-stages), i20s
# (run-gen-i20, the same clocks), i20c (run-gen-i20 with --moe-dispatch-clock); order 1 (i15s, i20s, i20c) x 5, order 2
# reversed x 5, each run pinned to the P-cores (taskset -c 0-7) inside the 1200% CPU cap. Waits for an idle card (the
# lock free, no compute app, 40 GiB MemAvailable) up to 48 h, then holds /tmp/memra-5090.lock for the 30 runs. Then
# the reader into the cell. Detached so it outlives the calling shell. Never signals another process.
# Environment seams for the dry check only: D83_BINS, D83_R, D83_LOCK, D83_ART, D83_WRAP (a command prefix replacing
# the systemd scope), D83_IDLE=skip.
# usage: nohup setsid bash rtx5090-queue-v17-20260926.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
BINS=${D83_BINS:-$T/target/c-bins}
R=${D83_R:-$L/rtx5090-day83}
LOCK=${D83_LOCK:-/tmp/memra-5090.lock}
ART=${D83_ART:-/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
PCORES=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo 0-7)
EV=$R/split20/ev
log() { echo "$(date -u +%FT%TZ) $*"; }
log "queue v17 start at $(git -C "$T" rev-parse --short HEAD)"
[ -f "$R/split20.done" ] && { log "split20 already done"; exit 0; }
for b in run-gen-i15 run-gen-i20; do [ -x "$BINS/$b" ] || { log "missing $b, stopping"; exit 1; }; done
mkdir -p "$EV"
printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.tsv -whitespace\n*.sha256 -whitespace\n' > "$R/.gitattributes"
if [ "${D83_IDLE:-wait}" != skip ]; then
    n=0
    while :; do
        if flock -n "$LOCK" true 2>/dev/null \
           && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
           && [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge $((40 * 1024 * 1024)) ]; then
            break
        fi
        n=$((n + 1)); [ $n -ge 34560 ] && { log "card never idle in 48 h"; exit 1; }
        [ $((n % 360)) -eq 0 ] && log "waiting for an idle card ($((n / 720)) h)"
        sleep 5
    done
fi
exec 9> "$LOCK"
flock -w 600 9 || { log "lock not taken in 600 s"; exit 1; }
log "lock held; runs start (P-cores $PCORES)"
git -C "$T" rev-parse HEAD > "$EV/tree.sha"
sha256sum "$BINS/run-gen-i15" "$BINS/run-gen-i20" > "$EV/binary.sha256"
stat -c '%n %s %Y' "$ART" > "$EV/artifact.stat"
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "p_cores=$PCORES"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; } > "$EV/host.txt" 2>&1
: > "$EV/marks.tsv"
snap() { { date -u +%FT%T.%3NZ; cat /proc/loadavg; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
           nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1
           grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$EV/$1.snap"; }
arm() { # $1 i15s|i20s|i20c  $2 label
    local bin=run-gen-i20 clocks=(--moe-dispatch-clock --expert-bank-stages)
    case $1 in
        i15s) bin=run-gen-i15 ;;
        i20c) clocks=(--moe-dispatch-clock) ;;
    esac
    snap "$2.before"
    printf '%s\t%s start\n' "$(date -u +%FT%T.%3NZ)" "$2" >> "$EV/marks.tsv"
    if [ -n "${D83_WRAP:-}" ]; then
        # shellcheck disable=SC2086
        $D83_WRAP taskset -c "$PCORES" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" \
            55 88 13 --experts-via-tier --expert-bank-host-bytes=17179869184 "${clocks[@]}" > "$EV/$2.log" 2>&1
    else
        systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G taskset -c "$PCORES" env MEMRA_MOE_RESIDENT=0 \
            MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" 55 88 13 --experts-via-tier \
            --expert-bank-host-bytes=17179869184 "${clocks[@]}" > "$EV/$2.log" 2>&1
    fi
    local rc=$?
    echo "$rc" > "$EV/$2.exit"
    printf '%s\t%s end rc=%s\n' "$(date -u +%FT%T.%3NZ)" "$2" "$rc" >> "$EV/marks.tsv"
    snap "$2.after"
}
for i in 1 2 3 4 5; do arm i15s "o1-i15s-r$i"; arm i20s "o1-i20s-r$i"; arm i20c "o1-i20c-r$i"; done
for i in 1 2 3 4 5; do arm i20c "o2-i20c-r$i"; arm i20s "o2-i20s-r$i"; arm i15s "o2-i15s-r$i"; done
flock -u 9
log "runs done; lock released"
/usr/bin/python3 "$L/day83-read.py" "$EV" --arms i15s,i20s,i20c --rig rtx5090 --check > "$R/split20/reading.log" 2>&1
log "reader rc=$?"
touch "$R/split20.done"
log "queue v17 done"
