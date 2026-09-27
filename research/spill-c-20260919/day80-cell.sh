#!/usr/bin/env bash
# Day 80 cells `regpool` and `regtime` (lane/spill-c-20260919, research/spill-c-20260919/DAY80.md section 1,
# registered before this script): OWED C12, the registered pool. One binary p80.
# regpool: every run under induce-b's fragmentation (DAY74 section 4), DAY73's sampler and the phase probes; arms refi
# (REF), di (the door), dri (the door with --expert-bank-pool-registered); order 1 (refi, di, dri) x 4, order 2
# reversed x 4; then two census runs (c1-di, c1-dri) with day78-pages.py at the gate on the run's own process.
# regtime: no fragmentation, no sampler; arms ref, d, dr, order 1 (ref, d, dr) x 5, order 2 reversed x 5.
# usage: day80-cell.sh regpool|regtime <lockfd>
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
host_state() { # $1 label: the host's static CPU and kernel settings (read only)
    {
        echo "kernel $(uname -r)"
        echo "clocksource $(cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>&1)"
        echo "amd_pstate $(cat /sys/devices/system/cpu/amd_pstate/status 2>&1)"
        echo "thp $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>&1)"
        echo "cpuidle_driver $(cat /sys/devices/system/cpu/cpuidle/current_driver 2>&1)"
        for f in /proc/sys/vm/compaction_proactiveness /proc/sys/vm/compact_unevictable_allowed \
                 /proc/sys/vm/extfrag_threshold /proc/sys/vm/watermark_scale_factor /proc/sys/vm/watermark_boost_factor \
                 /proc/sys/vm/min_free_kbytes /proc/sys/vm/numa_stat /proc/sys/vm/zone_reclaim_mode \
                 /sys/kernel/mm/transparent_hugepage/defrag /sys/kernel/mm/transparent_hugepage/khugepaged/defrag \
                 /sys/kernel/mm/transparent_hugepage/khugepaged/pages_to_scan /proc/sys/kernel/numa_balancing; do
            echo "$f $(cat "$f" 2>&1)"
        done
        grep -E 'MemTotal|MemFree|MemAvailable|Unevictable|Mlocked|AnonHugePages|HugePages_Total|CmaTotal' /proc/meminfo
        for c in /sys/devices/system/cpu/cpu[0-9]*; do
            p=$c/cpufreq
            echo "$(basename "$c") driver=$(cat "$p/scaling_driver" 2>/dev/null) governor=$(cat "$p/scaling_governor" 2>/dev/null) epp=$(cat "$p/energy_performance_preference" 2>/dev/null) min=$(cat "$p/scaling_min_freq" 2>/dev/null) max=$(cat "$p/scaling_max_freq" 2>/dev/null)"
        done
        cat /proc/interrupts
    } > "$EV/host-$1.txt" 2>&1
}
case $cell in
regtime)
    sha256sum "$D40_BINS/run-gen-p80" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    tarm() { # $1 ref|d|dr  $2 label
        local pre=() door=(--experts-via-tier --expert-bank-host-bytes=17179869184)
        case $1 in
            ref) pre=(MEMRA_MOE_PREFETCH=1); door=() ;;
            dr) door+=(--expert-bank-pool-registered) ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$D40_BINS/run-gen-p80" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do tarm ref "o1-ref-r$i"; tarm d "o1-d-r$i"; tarm dr "o1-dr-r$i"; done
    for i in 1 2 3 4 5; do tarm dr "o2-dr-r$i"; tarm d "o2-d-r$i"; tarm ref "o2-ref-r$i"; done
    echo "regtime cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
regpool)
    sha256sum "$D40_BINS/run-gen-p80" | tee "$EV/binary.sha256"
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
    host_state before
    # The counters check: one counted chain in a separate process under the run's pin; the flag goes to the runs
    # only when it printed rdpru=ok.
    taskset -c "$one" "$D40_BINS/run-gen-p80" --cpu-probe-counters-check > "$EV/counters-check.txt" 2>&1
    echo "rc=$?" >> "$EV/counters-check.txt"
    counters=()
    if grep -q ' rdpru=ok ' "$EV/counters-check.txt"; then counters=(--cpu-probe-counters); fi
    echo "counters=${counters[*]:-off}" | tee "$EV/counters.txt"
    side=()
    [ "$pick" = none ] || side=(taskset -c "$pick")
    : > "$EV/sched.tsv"
    "${side[@]}" python3 "$D40_TREE/research/spill-c-20260919/day73-sampler.py" "$EV/sched.tsv" &
    sampler=$!
    # The inducer's size (DAY74 section 4): X = MemFree - 2 GiB, the induced arms only when X / 2 >= 48 GiB.
    free_gib=$(awk '/MemFree/ {printf "%d", $2 / 1048576}' /proc/meminfo)
    F=$(( free_gib - 2 ))
    induce=1; [ $(( F / 2 )) -lt 48 ] && induce=0
    # D74_DRY_F: the control-flow dry check's small inducer (day74-cpu/dry-check-cell.sh); the driver never sets it.
    [ -n "${D74_DRY_F:-}" ] && { F=$D74_DRY_F; induce=1; }
    echo "memfree_gib=$free_gib F_gib=$F induce=$induce" | tee "$EV/inducer.txt"
    : > "$EV/compactor.tsv"
    : > "$EV/rewarm.tsv"
    # The run-gen process below this cell's shell: a run-gen-p80 whose ancestors include $$.
    own_run_gen() {
        local p q
        for p in $(ps -eo pid,comm | awk '$2 == "run-gen-p80" {print $1}'); do
            q=$p
            while [ -n "$q" ] && [ "$q" -gt 1 ]; do
                [ "$q" = "$$" ] && { echo "$p"; return 0; }
                q=$(awk '{print $4}' "/proc/$q/stat" 2>/dev/null)
            done
        done
        return 1
    }
    arm() { # $1 refi|di|dri  $2 label
        local pre=() door=(--experts-via-tier --expert-bank-host-bytes=17179869184) kind=$1 label=$2 cpid='' w='' before
        case $kind in
            refi) pre=(MEMRA_MOE_PREFETCH=1); door=() ;;
            dri) door+=(--expert-bank-pool-registered) ;;
        esac
        # DAY74 section 4: the artifact reread through the page cache before every run.
        local t0; t0=$(date +%s%N); cat "$D40_ART" > /dev/null
        printf '%s\t%s\t%d ms\n' "$(date -u +%T.%3N)" "$label" $(( ($(date +%s%N) - t0) / 1000000 )) >> "$EV/rewarm.tsv"
        if true; then
            [ "$induce" = 1 ] || { echo "$label not run (induce=0)" >> "$EV/inducer.txt"; return 0; }
            printf '%s\t%s\n' "$(date -u +%T.%3N)" "$label" >> "$EV/compactor.tsv"
            before=$(grep -c $'\tready\t' "$EV/compactor.tsv")
            "${side[@]}" python3 "$D40_TREE/research/spill-c-20260919/day74-compactor.py" "$EV/compactor.tsv" "$F" &
            cpid=$!
            for _ in $(seq 1 3000); do [ "$(grep -c $'\tready\t' "$EV/compactor.tsv")" -gt "$before" ] && break; sleep 0.1; done
            ( for _ in $(seq 1 12000); do
                  if grep -q "phase=gate" "$EV/$label.log" 2>/dev/null; then
                      kill -USR1 "$cpid"
                      # DAY78: the census runs' page census at the gate, on this cell's own run-gen.
                      if [ "${label:0:1}" = c ]; then
                          p=$(own_run_gen)
                          [ -n "$p" ] && "${side[@]}" python3 "$D40_TREE/research/spill-c-20260919/day78-pages.py" \
                              "$p" "$EV/$label.pages.tsv" --gap 0.5 > "$EV/$label.pages.log" 2>&1
                      fi
                      exit 0
                  fi
                  sleep 0.05
              done ) &
            w=$!
        fi
        run_gen "$label" taskset -c "$one" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$D40_BINS/run-gen-p80" "$D40_ART" 55 88 13 "${door[@]}" --cpu-probe --cpu-probe-phases "${counters[@]}"
        if [ -n "$cpid" ]; then
            kill "$w" 2>/dev/null; kill -TERM "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
        fi
    }
    for i in 1 2 3 4; do arm refi "o1-refi-r$i"; arm di "o1-di-r$i"; arm dri "o1-dri-r$i"; done
    for i in 1 2 3 4; do arm dri "o2-dri-r$i"; arm di "o2-di-r$i"; arm refi "o2-refi-r$i"; done
    for k in di dri; do arm "$k" "c1-$k-r1"; done
    kill "$sampler" 2>/dev/null
    host_state after
    echo "regpool cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
