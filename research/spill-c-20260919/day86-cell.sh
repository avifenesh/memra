#!/usr/bin/env bash
# Day 86 cell `slow86` (lane/spill-c-20260919, research/spill-c-20260919/DAY86.md section 1, registered before this
# script): whether the door's recent cuts make C12's slow state more frequent on a long-running 9950X host. Arms: ref
# (run-gen-i21, MEMRA_MOE_PREFETCH=1, no door), i20, i21, i22 (the door), i22r (run-gen-i22 with
# --expert-bank-pool-registered); order 1 (ref, i20, i21, i22, i22r) x 5, order 2 reversed x 5; without run-gen-i22
# the i22 and i22r arms are dropped. Every run boundary snapshot adds /proc/vmstat's compact_*, pgmigrate_* and
# thp_fault_* counters and /proc/buddyinfo. No profiler. One collector lock hold. The day-18 overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day86-cell.sh slow86 <lockfd>
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
      grep -E 'MemAvailable|SwapFree' /proc/meminfo
      echo "## vmstat"; grep -E '^(compact_|pgmigrate_|thp_fault_)' /proc/vmstat
      echo "## buddyinfo"; cat /proc/buddyinfo; } > "$EV/$1.snap"
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
case $cell in
slow86)
    bins=(run-gen-i20 run-gen-i21); arms=(ref i20 i21)
    if [ -x "$D40_BINS/run-gen-i22" ]; then bins+=(run-gen-i22); arms+=(i22 i22r); fi
    for b in "${bins[@]}"; do sha256sum "$D40_BINS/$b"; done | tee "$EV/binary.sha256"
    echo "arms: ${arms[*]}" | tee "$EV/arms.txt"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    { uptime; cat /proc/loadavg; } > "$EV/host-uptime.txt" 2>&1
    arm() { # $1 ref|i20|i21|i22|i22r  $2 label
        local pre=() door=(--experts-via-tier --expert-bank-host-bytes=17179869184) bin
        case $1 in
            ref) pre=(MEMRA_MOE_PREFETCH=1); door=(); bin=run-gen-i21 ;;
            i20) bin=run-gen-i20 ;;
            i21) bin=run-gen-i21 ;;
            i22) bin=run-gen-i22 ;;
            i22r) bin=run-gen-i22; door+=(--expert-bank-pool-registered) ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$D40_BINS/$bin" "$D40_ART" 55 88 13 "${door[@]}"
    }
    reversed=(); for ((k = ${#arms[@]} - 1; k >= 0; k--)); do reversed+=("${arms[k]}"); done
    for i in 1 2 3 4 5; do for a in "${arms[@]}"; do arm "$a" "o1-$a-r$i"; done; done
    for i in 1 2 3 4 5; do for a in "${reversed[@]}"; do arm "$a" "o2-$a-r$i"; done; done
    echo "slow86 cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
