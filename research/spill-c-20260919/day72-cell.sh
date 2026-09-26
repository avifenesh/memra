#!/usr/bin/env bash
# Day 72 cell `gap15` (lane/spill-c-20260919, research/spill-c-20260919/DAY72.md section 1, registered before this
# script): the door's remaining gap to REF re-attributed at I15. One collector lock hold, one binary (run-gen-i15)
# for every arm. Part A, DAY60's arms: ref (MEMRA_MOE_PREFETCH=1), refc (ref with --moe-dispatch-clock), on
# (--experts-via-tier --expert-bank-host-bytes=17179869184), onc (on with --moe-dispatch-clock); order 1 (ref, refc,
# on, onc) x 5, order 2 reversed x 5. Part B, after Part A: ref and on under Nsight Systems (--trace=cuda, no
# sampling, no context switches), order ref, on, on, ref, each report exported to SQLite. The day-18 overlap
# environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day72-cell.sh gap15 <lockfd>
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
gap15)
    sha256sum "$D40_BINS/run-gen-i15" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    arm() { # $1 ref|refc|on|onc  $2 label
        local pre=() door=()
        case $1 in
            ref) pre=(MEMRA_MOE_PREFETCH=1) ;;
            refc) pre=(MEMRA_MOE_PREFETCH=1); door=(--moe-dispatch-clock) ;;
            on) door=(--experts-via-tier --expert-bank-host-bytes=17179869184) ;;
            onc) door=(--experts-via-tier --expert-bank-host-bytes=17179869184 --moe-dispatch-clock) ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" "$D40_BINS/run-gen-i15" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do arm ref "o1-ref-r$i"; arm refc "o1-refc-r$i"; arm on "o1-on-r$i"; arm onc "o1-onc-r$i"; done
    for i in 1 2 3 4 5; do arm onc "o2-onc-r$i"; arm on "o2-on-r$i"; arm refc "o2-refc-r$i"; arm ref "o2-ref-r$i"; done
    # Part B: the profiler, from the toolkit the build used when PATH has none.
    NSYS=$(command -v nsys || { [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nsys" ] && echo "${CUDA_HOME:-/usr/local/cuda}/bin/nsys"; } || true)
    if [ -z "$NSYS" ]; then
        echo "nsys=none" | tee "$EV/nsys.txt"
    else
        { echo "nsys=$NSYS"; "$NSYS" --version; } 2>&1 | tee "$EV/nsys.txt"
        prof() { # $1 ref|on  $2 label
            local pre=() door=()
            case $1 in
                ref) pre=(MEMRA_MOE_PREFETCH=1) ;;
                on) door=(--experts-via-tier --expert-bank-host-bytes=17179869184) ;;
            esac
            run_gen "$2" "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
                -o "$EV/$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "${pre[@]}" \
                "$D40_BINS/run-gen-i15" "$D40_ART" 55 88 13 "${door[@]}"
            "$NSYS" export --type sqlite --force-overwrite=true -o "$EV/$2.sqlite" "$EV/$2.nsys-rep" \
                > "$EV/$2.export.log" 2>&1
            echo "rc=$?" >> "$EV/$2.export.log"
        }
        prof ref p-ref-r1; prof on p-on-r1; prof on p-on-r2; prof ref p-ref-r2
    fi
    echo "gap15 cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
