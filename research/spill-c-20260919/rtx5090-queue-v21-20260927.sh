#!/usr/bin/env bash
# Lane C RTX 5090 queue v21 (2026-09-27): DAY89 section 2's local check and in-situ split of I23 and I24 beside their
# parent p88 (research/spill-c-20260919/DAY89.md, registered before this script). Part 1, the check: run-gen-p88 and
# run-gen-i24 with the door, no clocks, order p88 i24 i24 p88, into check/; read by day89-cpu/gpu-check-read.py (MATCH,
# the same tape and host demand sequence). Part 2, the split: arms p88s, i23s, i24s (both clocks: --moe-dispatch-clock
# --expert-bank-stages) and p88d, i23d, i24d (the dispatch clock alone); order 1 (p88s, i23s, i24s, p88d, i23d, i24d)
# x 5, order 2 reversed x 5, into split24/; read by day83-read.py --check (integrity) and --change, and by
# day89-cpu/split-read.py (the sizing rule). Every run pinned to the P-cores (taskset) inside the 1200% CPU cap; one
# hold of /tmp/memra-5090.lock for the 64 runs after the idle wait (the lock free, no compute app, 40 GiB MemAvailable,
# up to 48 h). Detached so it outlives the calling shell. Never signals another process.
# Environment seams for the dry check only: D89_BINS, D89_R, D89_LOCK, D89_ART, D89_WRAP (a command prefix replacing
# the systemd scope), D89_IDLE=skip.
# usage: nohup setsid bash rtx5090-queue-v21-20260927.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
BINS=${D89_BINS:-$T/target/c-bins}
R=${D89_R:-$L/rtx5090-day89}
LOCK=${D89_LOCK:-/tmp/memra-5090.lock}
ART=${D89_ART:-/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
PCORES=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo 0-7)
log() { echo "$(date -u +%FT%TZ) $*"; }
log "queue v21 start at $(git -C "$T" rev-parse --short HEAD)"
[ -f "$R/v21.done" ] && { log "v21 already done"; exit 0; }
for b in run-gen-p88 run-gen-i23 run-gen-i24; do [ -x "$BINS/$b" ] || { log "missing $b, stopping"; exit 1; }; done
mkdir -p "$R/check" "$R/split24/ev"
printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.tsv -whitespace\n*.sha256 -whitespace\n' > "$R/.gitattributes"
if [ "${D89_IDLE:-wait}" != skip ]; then
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
EV=$R/split24/ev
git -C "$T" rev-parse HEAD > "$EV/tree.sha"
sha256sum "$BINS/run-gen-p88" "$BINS/run-gen-i23" "$BINS/run-gen-i24" | tee "$EV/binary.sha256" > "$R/check/binary.sha256"
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
    if [ -n "${D89_WRAP:-}" ]; then
        # shellcheck disable=SC2086
        $D89_WRAP taskset -c "$PCORES" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" \
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
for label in p88-a i24-a i24-b p88-b; do run "$R/check" "$label" "run-gen-${label%-*}"; done
both=(--moe-dispatch-clock --expert-bank-stages)
one=(--moe-dispatch-clock)
arm() { # $1 order  $2 arm  $3 run
    local bin=run-gen-${2%?}
    case $2 in
        *s) run "$EV" "$1-$2-r$3" "$bin" "${both[@]}" ;;
        *d) run "$EV" "$1-$2-r$3" "$bin" "${one[@]}" ;;
    esac
}
for i in 1 2 3 4 5; do for a in p88s i23s i24s p88d i23d i24d; do arm o1 "$a" "$i"; done; done
for i in 1 2 3 4 5; do for a in i24d i23d p88d i24s i23s p88s; do arm o2 "$a" "$i"; done; done
flock -u 9
log "runs done; lock released"
{ echo "# DAY89 local GPU check (the development host's RTX 5090 under its lock): the door at p88 (run-gen-p88 = 0155bc69f) and at I24 (run-gen-i24 = 1fd4b24c0), 32 tokens each, order p88, i24, i24, p88; raw logs in check/"
  cat "$R/check/binary.sha256"; /usr/bin/python3 "$L/day89-cpu/gpu-check-read.py" "$R/check" p88-a,i24-a,i24-b,p88-b; } > "$R/check/reading.log" 2>&1
crc=$?
log "check reader rc=$crc"
arms=p88s,i23s,i24s,p88d,i23d,i24d
{ /usr/bin/python3 "$L/day83-read.py" "$EV" --arms "$arms" --rig rtx5090 --check --change p88s,i23s
  /usr/bin/python3 "$L/day83-read.py" "$EV" --arms i23s,i24s --rig rtx5090 --change i23s,i24s | grep 'CHANGE'
  /usr/bin/python3 "$L/day89-cpu/split-read.py" "$EV" --rig rtx5090; } > "$R/split24/reading.log" 2>&1
src=$?
log "split reader rc=$src"
touch "$R/v21.done"
log "queue v21 done"
