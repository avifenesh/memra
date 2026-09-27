#!/usr/bin/env bash
# Lane C RTX 5090 queue v22 (2026-09-27): DAY92 section 3's local check and in-situ split of I25 (I25a, the trace
# behind --expert-bank-trace; I25b, the registry's owner check) beside p88 and I24 (research/spill-c-20260919/DAY92.md,
# registered before this script). Part 1, the check: run-gen-p88 and run-gen-i25 with --expert-bank-trace, order p88,
# i25t, i25t, p88, into check/, read by day89-cpu/gpu-check-read.py (MATCH, one tape, one host demand sequence); then
# one untraced run-gen-i25 run (i25u-a), whose tape must be p88-a's and whose trace must be empty. Part 2, the split:
# arms p88s, i24s, i25s (untraced) and i25t (--expert-bank-trace), each with both clocks (--moe-dispatch-clock
# --expert-bank-stages); order 1 (p88s, i24s, i25s, i25t) x 5, order 2 reversed x 5, into split25/; read by
# day83-read.py --check --traced p88s,i24s,i25t and by day92-cpu/split-read.py (the sizing rule, cumulative from p88s).
# Every run pinned to the P-cores (taskset) inside the 1200% CPU cap; one hold of /tmp/memra-5090.lock for the 45 runs
# after the idle wait (the lock free, no compute app, 40 GiB MemAvailable, up to 48 h). Detached so it outlives the
# calling shell. Never signals another process.
# Environment seams for the dry check only: D92_BINS, D92_R, D92_LOCK, D92_ART, D92_WRAP (a command prefix replacing
# the systemd scope), D92_IDLE=skip.
# usage: nohup setsid bash rtx5090-queue-v22-20260927.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
L=$T/research/spill-c-20260919
BINS=${D92_BINS:-$T/target/c-bins}
R=${D92_R:-$L/rtx5090-day92}
LOCK=${D92_LOCK:-/tmp/memra-5090.lock}
ART=${D92_ART:-/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
PCORES=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo 0-7)
log() { echo "$(date -u +%FT%TZ) $*"; }
log "queue v22 start at $(git -C "$T" rev-parse --short HEAD)"
[ -f "$R/v22.done" ] && { log "v22 already done"; exit 0; }
for b in run-gen-p88 run-gen-i24 run-gen-i25; do [ -x "$BINS/$b" ] || { log "missing $b, stopping"; exit 1; }; done
mkdir -p "$R/check" "$R/split25/ev"
printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.tsv -whitespace\n*.sha256 -whitespace\n' > "$R/.gitattributes"
if [ "${D92_IDLE:-wait}" != skip ]; then
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
EV=$R/split25/ev
git -C "$T" rev-parse HEAD > "$EV/tree.sha"
sha256sum "$BINS/run-gen-p88" "$BINS/run-gen-i24" "$BINS/run-gen-i25" | tee "$EV/binary.sha256" > "$R/check/binary.sha256"
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
    if [ -n "${D92_WRAP:-}" ]; then
        # shellcheck disable=SC2086
        $D92_WRAP taskset -c "$PCORES" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" \
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
run "$R/check" p88-a run-gen-p88
run "$R/check" i25t-a run-gen-i25 --expert-bank-trace
run "$R/check" i25t-b run-gen-i25 --expert-bank-trace
run "$R/check" p88-b run-gen-p88
run "$R/check" i25u-a run-gen-i25
both=(--moe-dispatch-clock --expert-bank-stages)
arm() { # $1 order  $2 arm  $3 run
    local bin=run-gen-${2%?}
    case $2 in
        *t) run "$EV" "$1-$2-r$3" "$bin" "${both[@]}" --expert-bank-trace ;;
        *) run "$EV" "$1-$2-r$3" "$bin" "${both[@]}" ;;
    esac
}
for i in 1 2 3 4 5; do for a in p88s i24s i25s i25t; do arm o1 "$a" "$i"; done; done
for i in 1 2 3 4 5; do for a in i25t i25s i24s p88s; do arm o2 "$a" "$i"; done; done
flock -u 9
log "runs done; lock released"
( echo "# DAY92 local GPU check (the development host's RTX 5090 under its lock): the door at p88 (run-gen-p88 = 0155bc69f) and at I25 (run-gen-i25 = ee41ede8f) with --expert-bank-trace, 32 tokens each, order p88, i25t, i25t, p88, then one untraced i25 run; raw logs in check/"
  cat "$R/check/binary.sha256"; /usr/bin/python3 "$L/day89-cpu/gpu-check-read.py" "$R/check" p88-a,i25t-a,i25t-b,p88-b
  gc=$?
  u=$R/check/i25u-a.log
  ut=$(grep -m1 '^tokens: ' "$u"); pt=$(grep -m1 '^tokens: ' "$R/check/p88-a.log")
  ul=$(grep -c 'expert-host-slru\] key=' "$u")
  urc=$(cat "$R/check/i25u-a.exit")
  if [ "$urc" = 0 ] && grep -q 'MATCH' "$u" && [ -n "$ut" ] && [ "$ut" = "$pt" ] && [ "$ul" = 0 ]; then
      echo "DAY92 UNTRACED i25u-a rc=0 MATCH the same tape as p88-a trace_lines=0 -> PASS"
  else
      echo "DAY92 UNTRACED i25u-a rc=$urc tape_same=$([ "$ut" = "$pt" ] && echo yes || echo no) trace_lines=$ul -> FAIL"
      gc=1
  fi
  exit "$gc" ) > "$R/check/reading.log" 2>&1
crc=$?
log "check reader rc=$crc"
arms=p88s,i24s,i25s,i25t
{ /usr/bin/python3 "$L/day83-read.py" "$EV" --arms "$arms" --rig rtx5090 --check --traced p88s,i24s,i25t --change i24s,i25s
  /usr/bin/python3 "$L/day92-cpu/split-read.py" "$EV" --rig rtx5090; } > "$R/split25/reading.log" 2>&1
src=$?
log "split reader rc=$src"
touch "$R/v22.done"
log "queue v22 done"
