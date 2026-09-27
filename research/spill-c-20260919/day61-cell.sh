#!/usr/bin/env bash
# Day 61 cell `i11` (lane/spill-c-20260919, research/spill-c-20260919/DAY61.md section 2, registered before the
# improvements' code): I11 and I12 against the door they tune and against REF. One collector lock hold. Arms: ref
# (MEMRA_MOE_PREFETCH=1, run-gen-c60), on (the door at the full-bank host budget, run-gen-c60), i11 (the same door,
# run-gen-i11: I11 changes 1 to 5), i12 (the same door, run-gen-i12: I11 plus I12). Order 1 (ref, on, i11, i12) x 5,
# order 2 reversed x 5; the day-18 overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day61-cell.sh i11 <lockfd>
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
i11)
    sha256sum "$D40_BINS/run-gen-c60" "$D40_BINS/run-gen-i11" "$D40_BINS/run-gen-i12" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    arm() { # $1 ref|on|i11|i12  $2 label
        local pre=() door=() bin=$D40_BINS/run-gen-c60
        case $1 in
            ref) pre=(MEMRA_MOE_PREFETCH=1) ;;
            on) door=(--experts-via-tier --expert-bank-host-bytes=17179869184) ;;
            i11) door=(--experts-via-tier --expert-bank-host-bytes=17179869184); bin=$D40_BINS/run-gen-i11 ;;
            i12) door=(--experts-via-tier --expert-bank-host-bytes=17179869184); bin=$D40_BINS/run-gen-i12 ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$bin" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do arm ref "o1-ref-r$i"; arm on "o1-on-r$i"; arm i11 "o1-i11-r$i"; arm i12 "o1-i12-r$i"; done
    for i in 1 2 3 4 5; do arm i12 "o2-i12-r$i"; arm i11 "o2-i11-r$i"; arm on "o2-on-r$i"; arm ref "o2-ref-r$i"; done
    echo "i11 cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
