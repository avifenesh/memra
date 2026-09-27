#!/usr/bin/env bash
# Day 51 cells (lane/spill-c-20260919, research/spill-c-20260919/DAY51.md, pre-registered before any improvement rung
# was measured): the MoE slot cache door's deciding sitting on the final tree. One collector lock hold per cell.
#   hashlock  G1: the door on a non-approved single-shard artifact refuses on its SHA lock; the control MATCHes.
#   spec      G2: run-spec K=1..8 under the tuned door, the registered shape (day 11's, default GPU budget) and the
#             two section-1a shapes: day 18's pressure (9,986 slots) and day 11's eight-slot GPU budget.
#   decide    the timing cell: off (legacy naked default), on (--experts-via-tier at the full-bank budget, no stage
#             clock), ref (legacy MEMRA_MOE_PREFETCH=1); order 1 (off, on, ref) x 5, order 2 (ref, on, off) x 5.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir (run-gen-final, run-spec-final),
# D40_TREE worktree, D40_ART the approved artifact, D40_ART_OTHER a non-approved single-shard artifact, D40_LOCK,
# D51_MOE_ENV the hashlock cell's MoE env as on day 18 (empty on the RTX 5090, MEMRA_MOE_RESIDENT=0 on the target).
# usage: day51-cell.sh <hashlock|spec|decide> <lockfd>
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
hashlock)
    : "${D40_ART_OTHER:?}"
    sha256sum "$D40_BINS/run-gen-final" | tee "$EV/binary.sha256"
    sha256sum "$D40_ART_OTHER" | tee "$EV/artifact-other.sha256"
    # shellcheck disable=SC2086
    run_gen door env ${D51_MOE_ENV:-} MEMRA_NGEN=8 "$D40_BINS/run-gen-final" "$D40_ART_OTHER" 55 88 13 --experts-via-tier
    # shellcheck disable=SC2086
    run_gen control env ${D51_MOE_ENV:-} MEMRA_NGEN=8 "$D40_BINS/run-gen-final" "$D40_ART_OTHER" 55 88 13
    echo "hashlock cell done: door rc=$(cat "$EV/door.exit") control rc=$(cat "$EV/control.exit")"
    ;;
spec)
    sha256sum "$D40_BINS/run-spec-final" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    host=(--experts-via-tier --expert-bank-host-bytes=17179869184)
    run_gen spec env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 "$D40_BINS/run-spec-final" "$D40_ART" 55 88 13 "${host[@]}"
    run_gen spec-pressure env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$D40_BINS/run-spec-final" "$D40_ART" 55 88 13 "${host[@]}"
    run_gen spec-exact8 env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 "$D40_BINS/run-spec-final" "$D40_ART" 55 88 13 "${host[@]}" --expert-bank-gpu-bytes=6881344
    echo "spec cell done: spec rc=$(cat "$EV/spec.exit") pressure rc=$(cat "$EV/spec-pressure.exit") exact8 rc=$(cat "$EV/spec-exact8.exit")"
    ;;
decide|decide-b)
    # DAY51 section 1c: decide-b is the same cell, read with the corrected trace term.
    sha256sum "$D40_BINS/run-gen-final" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    arm() { # $1 off|on|ref  $2 label
        local pre=() door=()
        case $1 in
            on) door=(--experts-via-tier --expert-bank-host-bytes=17179869184) ;;
            ref) pre=(MEMRA_MOE_PREFETCH=1) ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$D40_BINS/run-gen-final" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do arm off "o1-off-r$i"; arm on "o1-on-r$i"; arm ref "o1-ref-r$i"; done
    for i in 1 2 3 4 5; do arm ref "o2-ref-r$i"; arm on "o2-on-r$i"; arm off "o2-off-r$i"; done
    echo "$cell cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
