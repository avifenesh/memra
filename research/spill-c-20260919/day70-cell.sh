#!/usr/bin/env bash
# Day 70 cell `sched` (lane/spill-c-20260919, research/spill-c-20260919/DAY70.md section 1, registered before this
# script): OWED C12, REF and the door (I15) from the binary p68 on DAY65's ONE pin, --cpu-probe and --cpu-probe-phases
# on every run, and one sampler (day70-sampler.py: every CPU's /proc/stat and the owner thread's schedstat every
# 250 ms, the host's moving threads every 1 s) pinned to a CPU outside the pin and its hardware-thread siblings.
# Order 1 (ref, i15) x 10, order 2 (i15, ref) x 10; the day-18 overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day70-cell.sh sched <lockfd>
set -uo pipefail
cell=$1; fd=$2
: "${D40_R:?}" "${D40_BINS:?}" "${D40_TREE:?}" "${D40_LOCK:?}" "${D40_ART:?}"
EV=$D40_R/$cell/ev
mkdir -p "$EV"
cd "$D40_TREE" || exit 1
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$D40_LOCK" --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/tree.sha"
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
: > "$EV/marks.tsv"
# stamp: prefix every line of stdin with a UTC ms timestamp (line arrival time); nothing is dropped.
stamp() { python3 -c '
import sys, datetime
for line in sys.stdin.buffer:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    sys.stdout.buffer.write(ts.encode() + b"\t" + line); sys.stdout.buffer.flush()
' > "$1"; }
snap() { # $1 label: the card's compute apps and host memory at a run boundary (never acted on)
    { date -u +%FT%T.%3NZ; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
      grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$EV/$1.snap"
}
run_gen() { # $1 label  $2.. argv (env words first)
    local label=$1; shift
    snap "$label.before"
    mark "$label start"
    "$@" 2>&1 | stamp "$EV/$label.log"
    local rc=${PIPESTATUS[0]}
    echo "$rc" > "$EV/$label.exit"
    mark "$label end rc=$rc"
    snap "$label.after"
    return 0
}
# expand: a CPU list ("0-7,16-23") as one number per line
expand() { tr ',' '\n' <<< "$1" | awk -F- '{ if (NF == 2) { for (i = $1; i <= $2; i++) print i } else print $1 }'; }
case $cell in
sched)
    sha256sum "$D40_BINS/run-gen-p68" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    {
        lscpu
        for c in /sys/devices/system/cpu/cpu[0-9]*; do
            echo "$(basename "$c") l3=$(cat "$c/cache/index3/shared_cpu_list" 2>/dev/null) siblings=$(cat "$c/topology/thread_siblings_list" 2>/dev/null)"
        done
    } > "$EV/topology.txt" 2>&1
    home=$(cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list)
    one=$(expand "$home" | sort -n | head -12 | paste -sd, -)
    one_set=$(expand "$one")
    # The sampler's CPU: the highest CPU outside the pin whose hardware-thread siblings are all outside it too.
    pick=none
    for c in $(for d in /sys/devices/system/cpu/cpu[0-9]*; do echo "${d##*cpu}"; done | sort -rn); do
        clash=0
        for s in $(expand "$(cat "/sys/devices/system/cpu/cpu$c/topology/thread_siblings_list")"); do
            grep -qx "$s" <<< "$one_set" && clash=1
        done
        if [ "$clash" = 0 ]; then pick=$c; break; fi
    done
    printf 'home_l3=%s\none=%s\nsampler_cpu=%s\n' "$home" "$one" "$pick" | tee "$EV/pins.txt"
    : > "$EV/sched.tsv"
    if [ "$pick" = none ]; then
        python3 "$D40_TREE/research/spill-c-20260919/day70-sampler.py" "$EV/sched.tsv" &
    else
        taskset -c "$pick" python3 "$D40_TREE/research/spill-c-20260919/day70-sampler.py" "$EV/sched.tsv" &
    fi
    sampler=$!
    arm() { # $1 ref|i15  $2 label
        local pre=() door=(--experts-via-tier --expert-bank-host-bytes=17179869184)
        if [ "$1" = ref ]; then pre=(MEMRA_MOE_PREFETCH=1); door=(); fi
        run_gen "$2" taskset -c "$one" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$D40_BINS/run-gen-p68" "$D40_ART" 55 88 13 "${door[@]}" --cpu-probe --cpu-probe-phases
    }
    for i in $(seq 1 10); do arm ref "o1-ref-r$i"; arm i15 "o1-i15-r$i"; done
    for i in $(seq 1 10); do arm i15 "o2-i15-r$i"; arm ref "o2-ref-r$i"; done
    kill "$sampler" 2>/dev/null
    echo "sched cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
