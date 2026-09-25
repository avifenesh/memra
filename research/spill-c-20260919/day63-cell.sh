#!/usr/bin/env bash
# Day 63 cell `i13` (lane/spill-c-20260919, research/spill-c-20260919/DAY63.md section 3, registered before this
# script): I13 against the door it tunes and against REF, and I13's stage split on the card. One collector lock hold.
# Arms: ref (MEMRA_MOE_PREFETCH=1, run-gen-c60), i12 (the door at the full-bank host budget, run-gen-i12), i13 (the
# same door, run-gen-i13), i13s (i13 with --expert-bank-stages, read for its split only). Order 1 (ref, i12, i13,
# i13s) x 5, order 2 reversed x 5; the day-18 overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day63-cell.sh i13 <lockfd>
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
case $cell in
i13)
    sha256sum "$D40_BINS/run-gen-c60" "$D40_BINS/run-gen-i12" "$D40_BINS/run-gen-i13" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    arm() { # $1 ref|i12|i13|i13s  $2 label
        local pre=() door=(--experts-via-tier --expert-bank-host-bytes=17179869184) bin=$D40_BINS/run-gen-i13
        case $1 in
            ref) pre=(MEMRA_MOE_PREFETCH=1); door=(); bin=$D40_BINS/run-gen-c60 ;;
            i12) bin=$D40_BINS/run-gen-i12 ;;
            i13) ;;
            i13s) door+=(--expert-bank-stages) ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$bin" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do arm ref "o1-ref-r$i"; arm i12 "o1-i12-r$i"; arm i13 "o1-i13-r$i"; arm i13s "o1-i13s-r$i"; done
    for i in 1 2 3 4 5; do arm i13s "o2-i13s-r$i"; arm i13 "o2-i13-r$i"; arm i12 "o2-i12-r$i"; arm ref "o2-ref-r$i"; done
    echo "i13 cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
