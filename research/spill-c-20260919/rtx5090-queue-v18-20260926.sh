#!/usr/bin/env bash
# Lane C RTX 5090 queue v18 (2026-09-26): DAY84 section 1's local RTX 5090 check and in-situ split of I21 beside I20
# (research/spill-c-20260919/DAY84.md, registered before this script). Part 1, the check: run-gen-i20 and run-gen-i21
# with the door, no clocks, order i20 i21 i21 i20, into check/; read by day84-cpu/gpu-check-read.py (MATCH, the same
# tape and host demand sequence). Part 2, the split: arms i20s (run-gen-i20) and i21s (run-gen-i21), each with
# --moe-dispatch-clock --expert-bank-stages, order 1 (i20s, i21s) x 5, order 2 reversed x 5, into split21/; read by
# day83-read.py --check --change i20s,i21s, deciding nothing. Every run pinned to the P-cores (taskset) inside the
# 1200% CPU cap; one hold of /tmp/memra-5090.lock for the 24 runs after the idle wait (the lock free, no compute app,
# 40 GiB MemAvailable, up to 48 h). Detached so it outlives the calling shell. Never signals another process.
# Environment seams for the dry check only: D84_BINS, D84_R, D84_LOCK, D84_ART, D84_WRAP (a command prefix replacing
# the systemd scope), D84_IDLE=skip.
# usage: nohup setsid bash rtx5090-queue-v18-20260926.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
BINS=${D84_BINS:-$T/target/c-bins}
R=${D84_R:-$L/rtx5090-day84}
LOCK=${D84_LOCK:-/tmp/memra-5090.lock}
ART=${D84_ART:-/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
PCORES=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo 0-7)
log() { echo "$(date -u +%FT%TZ) $*"; }
log "queue v18 start at $(git -C "$T" rev-parse --short HEAD)"
[ -f "$R/v18.done" ] && { log "v18 already done"; exit 0; }
for b in run-gen-i20 run-gen-i21; do [ -x "$BINS/$b" ] || { log "missing $b, stopping"; exit 1; }; done
mkdir -p "$R/check" "$R/split21/ev"
printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.tsv -whitespace\n*.sha256 -whitespace\n' > "$R/.gitattributes"
if [ "${D84_IDLE:-wait}" != skip ]; then
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
EV=$R/split21/ev
git -C "$T" rev-parse HEAD > "$EV/tree.sha"
sha256sum "$BINS/run-gen-i20" "$BINS/run-gen-i21" | tee "$EV/binary.sha256" > "$R/check/binary.sha256"
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
    if [ -n "${D84_WRAP:-}" ]; then
        # shellcheck disable=SC2086
        $D84_WRAP taskset -c "$PCORES" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" \
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
for label in i20-a i21-a i21-b i20-b; do run "$R/check" "$label" "run-gen-${label%-*}"; done
clocks=(--moe-dispatch-clock --expert-bank-stages)
for i in 1 2 3 4 5; do run "$EV" "o1-i20s-r$i" run-gen-i20 "${clocks[@]}"; run "$EV" "o1-i21s-r$i" run-gen-i21 "${clocks[@]}"; done
for i in 1 2 3 4 5; do run "$EV" "o2-i21s-r$i" run-gen-i21 "${clocks[@]}"; run "$EV" "o2-i20s-r$i" run-gen-i20 "${clocks[@]}"; done
flock -u 9
log "runs done; lock released"
{ echo "# DAY84 local GPU check (the development host's RTX 5090 under its lock): the door at I20 (run-gen-i20 = 8efea3a54) and at I21 (run-gen-i21 = b555b4141), 32 tokens each, order i20, i21, i21, i20; raw logs in check/"
  cat "$R/check/binary.sha256"; /usr/bin/python3 "$L/day84-cpu/gpu-check-read.py" "$R/check"; } > "$R/check/reading.log" 2>&1
log "check reader rc=$?"
/usr/bin/python3 "$L/day83-read.py" "$EV" --arms i20s,i21s --rig rtx5090 --check --change i20s,i21s > "$R/split21/reading.log" 2>&1
log "split reader rc=$?"
touch "$R/v18.done"
log "queue v18 done"
