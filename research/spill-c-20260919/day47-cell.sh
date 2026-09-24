#!/usr/bin/env bash
# Day 47 cell `pinned` (lane/spill-c-20260919, research/spill-c-20260919/DAY47.md section 1, pre-registered before
# any I2 code): improvement I2, bounded pinned host slots. One collector lock hold. Arms: off (run-gen-i2, no door),
# i1 (run-gen-i1, the day-46 binary), i2 (run-gen-i2); both door arms with --expert-bank-stages and
# --expert-bank-host-bytes=17179869184. Order 1 (off, i1, i2) x 5, order 2 (i2, i1, off) x 5; the day-18 overlap
# environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day47-cell.sh pinned <lockfd>
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
pinned)
    sha256sum "$D40_BINS/run-gen-i1" "$D40_BINS/run-gen-i2" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    arm() { # $1 off|i1|i2  $2 label
        local bin=$D40_BINS/run-gen-i2 door=()
        case $1 in
            i1) bin=$D40_BINS/run-gen-i1; door=(--experts-via-tier --expert-bank-stages --expert-bank-host-bytes=17179869184) ;;
            i2) door=(--experts-via-tier --expert-bank-stages --expert-bank-host-bytes=17179869184) ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$bin" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do arm off "o1-off-r$i"; arm i1 "o1-i1-r$i"; arm i2 "o1-i2-r$i"; done
    for i in 1 2 3 4 5; do arm i2 "o2-i2-r$i"; arm i1 "o2-i1-r$i"; arm off "o2-off-r$i"; done
    echo "pinned cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
