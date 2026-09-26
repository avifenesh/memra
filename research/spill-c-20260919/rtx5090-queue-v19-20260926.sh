#!/usr/bin/env bash
# Lane C RTX 5090 queue v19 (2026-09-26): DAY85 section 3's local RTX 5090 check and in-situ split of I22 beside I21
# (research/spill-c-20260919/DAY85.md, registered before this script). Part 1, the check: run-gen-i21 and run-gen-i22
# with the door, no clocks, order i21 i22 i22 i21, into check/; read by day85-cpu/gpu-check-read.py (MATCH, the same
# tape and host demand sequence). Part 2, the split: arms i21s (run-gen-i21) and i22s (run-gen-i22), each with
# --moe-dispatch-clock --expert-bank-stages, order 1 (i21s, i22s) x 5, order 2 reversed x 5, into split22/; read by
# day83-read.py --check --change i21s,i22s, deciding nothing. Every run pinned to the P-cores (taskset) inside the
# 1200% CPU cap; one hold of /tmp/memra-5090.lock for the 24 runs after the idle wait (the lock free, no compute app,
# 40 GiB MemAvailable, up to 48 h). Detached so it outlives the calling shell. Never signals another process.
# Environment seams for the dry check only: D85_BINS, D85_R, D85_LOCK, D85_ART, D85_WRAP (a command prefix replacing
# the systemd scope), D85_IDLE=skip.
# usage: nohup setsid bash rtx5090-queue-v19-20260926.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
BINS=${D85_BINS:-$T/target/c-bins}
R=${D85_R:-$L/rtx5090-day85}
LOCK=${D85_LOCK:-/tmp/memra-5090.lock}
ART=${D85_ART:-/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
PCORES=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo 0-7)
log() { echo "$(date -u +%FT%TZ) $*"; }
log "queue v19 start at $(git -C "$T" rev-parse --short HEAD)"
[ -f "$R/v19.done" ] && { log "v18 already done"; exit 0; }
for b in run-gen-i21 run-gen-i22; do [ -x "$BINS/$b" ] || { log "missing $b, stopping"; exit 1; }; done
mkdir -p "$R/check" "$R/split22/ev"
printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.tsv -whitespace\n*.sha256 -whitespace\n' > "$R/.gitattributes"
if [ "${D85_IDLE:-wait}" != skip ]; then
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
EV=$R/split22/ev
git -C "$T" rev-parse HEAD > "$EV/tree.sha"
sha256sum "$BINS/run-gen-i21" "$BINS/run-gen-i22" | tee "$EV/binary.sha256" > "$R/check/binary.sha256"
stat -c '%n %s %Y' "$ART" > "$EV/artifact.stat"
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "p_cores=$PCORES"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; } > "$EV/host.txt" 2>&1
: > "$EV/marks.tsv"
snap() { # $1 dir  $2 label
    { date -u +%FT%T.%3NZ; cat /proc/loadavg; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
      nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1
      grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$1/$2.snap"; }
run() { # $1 dir  $2 label  $3 binary  $4.. extra flags
    local dir=$1 label=$2 bin=$3; shift 3
    snap "$dir" "$label.before"
    printf '%s\t%s start\n' "$(date -u +%FT%T.%3NZ)" "$label" >> "$EV/marks.tsv"
    if [ -n "${D85_WRAP:-}" ]; then
        # shellcheck disable=SC2086
        $D85_WRAP taskset -c "$PCORES" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" \
            55 88 13 --experts-via-tier --expert-bank-host-bytes=17179869184 "$@" > "$dir/$label.log" 2>&1
    else
        systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G taskset -c "$PCORES" env MEMRA_MOE_RESIDENT=0 \
            MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" 55 88 13 --experts-via-tier \
            --expert-bank-host-bytes=17179869184 "$@" > "$dir/$label.log" 2>&1
    fi
    local rc=$?
    echo "$rc" > "$dir/$label.exit"
    printf '%s\t%s end rc=%s\n' "$(date -u +%FT%T.%3NZ)" "$label" "$rc" >> "$EV/marks.tsv"
    snap "$dir" "$label.after"
}
for label in i21-a i22-a i22-b i21-b; do run "$R/check" "$label" "run-gen-${label%-*}"; done
clocks=(--moe-dispatch-clock --expert-bank-stages)
for i in 1 2 3 4 5; do run "$EV" "o1-i21s-r$i" run-gen-i21 "${clocks[@]}"; run "$EV" "o1-i22s-r$i" run-gen-i22 "${clocks[@]}"; done
for i in 1 2 3 4 5; do run "$EV" "o2-i22s-r$i" run-gen-i22 "${clocks[@]}"; run "$EV" "o2-i21s-r$i" run-gen-i21 "${clocks[@]}"; done
flock -u 9
log "runs done; lock released"
{ echo "# DAY85 local GPU check (the development host's RTX 5090 under its lock): the door at I21 (run-gen-i21 = b555b4141) and at I22 (run-gen-i22 = 4b378a064), 32 tokens each, order i21, i22, i22, i21; raw logs in check/"
  cat "$R/check/binary.sha256"; /usr/bin/python3 "$L/day85-cpu/gpu-check-read.py" "$R/check"; } > "$R/check/reading.log" 2>&1
log "check reader rc=$?"
/usr/bin/python3 "$L/day83-read.py" "$EV" --arms i21s,i22s --rig rtx5090 --check --change i21s,i22s > "$R/split22/reading.log" 2>&1
log "split reader rc=$?"
touch "$R/v19.done"
log "queue v19 done"
