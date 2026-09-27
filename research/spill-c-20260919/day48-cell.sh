#!/usr/bin/env bash
# Day 48 cell `small` (lane/spill-c-20260919, research/spill-c-20260919/DAY48.md section 1, pre-registered before
# any I8 or I5 code): improvements I8 (the validate memo) and I5 (the trace off the hot path) as two rungs. One
# collector lock hold. Arms: off (run-gen-i5, no door), i2 (run-gen-i2, the day-47 binary), i8 (run-gen-i8), i5
# (run-gen-i5); every door arm with --expert-bank-stages and --expert-bank-host-bytes=17179869184. Order 1 (off, i2,
# i8, i5) x 5, order 2 (i5, i8, i2, off) x 5; the day-18 overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day48-cell.sh small <lockfd>
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
small)
    sha256sum "$D40_BINS/run-gen-i2" "$D40_BINS/run-gen-i8" "$D40_BINS/run-gen-i5" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    arm() { # $1 off|i2|i8|i5  $2 label
        local bin=$D40_BINS/run-gen-i5 door=(--experts-via-tier --expert-bank-stages --expert-bank-host-bytes=17179869184)
        case $1 in
            off) door=() ;;
            i2) bin=$D40_BINS/run-gen-i2 ;;
            i8) bin=$D40_BINS/run-gen-i8 ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$bin" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do arm off "o1-off-r$i"; arm i2 "o1-i2-r$i"; arm i8 "o1-i8-r$i"; arm i5 "o1-i5-r$i"; done
    for i in 1 2 3 4 5; do arm i5 "o2-i5-r$i"; arm i8 "o2-i8-r$i"; arm i2 "o2-i2-r$i"; arm off "o2-off-r$i"; done
    echo "small cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
