#!/usr/bin/env bash
# Day 68 cell `eclock` (lane/spill-c-20260919, research/spill-c-20260919/DAY68.md section 1, registered before this
# script): OWED C12, REF and the door (I15) from one binary (label p68, which adds run-gen --cpu-probe-phases) on
# DAY65's ONE pin, --cpu-probe and --cpu-probe-phases on every run; every 250 ms the owner CPU's /proc/cpuinfo MHz
# (the clock the core ran at), its scaling_cur_freq (the requested clock) and the k10temp (or coretemp) temperatures;
# every 1 s the day-67 /proc/<pid>/sched counts. Order 1 (ref, i15) x 10, order 2 (i15, ref) x 10; the day-18
# overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day68-cell.sh eclock <lockfd>
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
# eclock_of: a CPU's /proc/cpuinfo MHz (the clock the core ran at) and its scaling_cur_freq (the requested one, kHz)
eclock_of() {
    local mhz req
    mhz=$(awk -v c="$1" '/^processor/ {p = $3} /^cpu MHz/ && p == c {print $4; exit}' /proc/cpuinfo)
    req=$(cat "/sys/devices/system/cpu/cpu$1/cpufreq/scaling_cur_freq" 2>/dev/null)
    printf '%s\t%s' "${mhz:-absent}" "${req:-absent}"
}
# temps: every k10temp (else coretemp) sensor as label=milli-degrees, one field
temps() {
    local h name out=""
    for h in /sys/class/hwmon/hwmon*; do
        name=$(cat "$h/name" 2>/dev/null)
        [ "$name" = k10temp ] || [ "$name" = coretemp ] || continue
        for t in "$h"/temp*_input; do
            [ -r "$t" ] || continue
            out="$out$(cat "${t%_input}_label" 2>/dev/null || basename "$t")=$(cat "$t"),"
        done
    done
    printf '%s' "${out:-absent}"
}
case $cell in
eclock)
    sha256sum "$D40_BINS/run-gen-p68" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    {
        lscpu
        for c in /sys/devices/system/cpu/cpu[0-9]*; do
            echo "$(basename "$c") l3=$(cat "$c/cache/index3/shared_cpu_list" 2>/dev/null)"
        done
        for f in scaling_driver scaling_governor energy_performance_preference scaling_min_freq scaling_max_freq; do
            echo "cpu0 cpufreq/$f=$(cat /sys/devices/system/cpu/cpu0/cpufreq/$f 2>/dev/null || echo absent)"
        done
        echo "cpufreq/boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo absent)"
        echo "amd_pstate/status=$(cat /sys/devices/system/cpu/amd_pstate/status 2>/dev/null || echo absent)"
        echo "thp/enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo absent)"
        echo "thp/defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo absent)"
    } > "$EV/topology.txt" 2>&1
    home=$(cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list)
    one=$(expand "$home" | sort -n | head -12 | paste -sd, -)
    printf 'home_l3=%s\none=%s\n' "$home" "$one" | tee "$EV/pins.txt"
    : > "$EV/eclock.tsv"
    ( while :; do
        t=$(date -u +%T.%3N)
        temp=$(temps)
        ps -eo pid,psr,comm | awk '$3 ~ /^run-gen/ {print $1, $2, $3}' | while read -r pid psr comm; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$pid" "$psr" "$comm" "$(eclock_of "$psr")" "$temp"
        done >> "$EV/eclock.tsv"
        sleep 0.25
      done ) &
    clock_sampler=$!
    : > "$EV/sched.tsv"
    ( while :; do
        t=$(date -u +%T.%3N)
        ps -eo pid,comm | awk '$2 ~ /^run-gen/ {print $1, $2}' | while read -r pid comm; do
            f=/proc/$pid/sched
            mig=$(awk '/^se.nr_migrations/ {print $3}' "$f" 2>/dev/null)
            sw=$(awk '/^nr_switches/ {print $3}' "$f" 2>/dev/null)
            vol=$(awk '/^nr_voluntary_switches/ {print $3}' "$f" 2>/dev/null)
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$pid" "$comm" "${mig:-absent}" "${sw:-absent}" "${vol:-absent}"
        done >> "$EV/sched.tsv"
        sleep 1
      done ) &
    sched_sampler=$!
    arm() { # $1 ref|i15  $2 label
        local pre=() door=(--experts-via-tier --expert-bank-host-bytes=17179869184)
        if [ "$1" = ref ]; then pre=(MEMRA_MOE_PREFETCH=1); door=(); fi
        run_gen "$2" taskset -c "$one" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$D40_BINS/run-gen-p68" "$D40_ART" 55 88 13 "${door[@]}" --cpu-probe --cpu-probe-phases
    }
    for i in $(seq 1 10); do arm ref "o1-ref-r$i"; arm i15 "o1-i15-r$i"; done
    for i in $(seq 1 10); do arm i15 "o2-i15-r$i"; arm ref "o2-ref-r$i"; done
    kill "$clock_sampler" "$sched_sampler" 2>/dev/null
    echo "eclock cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
