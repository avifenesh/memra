#!/usr/bin/env bash
# Day 52 cell `ladder` (lane/spill-c-20260919, research/spill-c-20260919/DAY52.md section 1, pre-registered before
# the target-card sitting): every improvement rung of the MoE slot cache door in one collector lock hold on the target
# card, so each rung's registered clauses (days 43 to 50) are read on this card from the same window
# (day52-views.py builds each day's arm set as a view and runs that day's reader unchanged).
# Arms: off (run-gen-final, no door); base (run-gen, the day-40 binary, default host budget); i6d (run-gen-i6,
# default budget); i6g (run-gen-i6) and i9g, fill, i1, i2, i8, i5, i7, i4 (run-gen-<rung>) at
# --expert-bank-host-bytes=17179869184; every door arm with --expert-bank-stages. Order 1 old to new x 5, order 2
# new to old x 5; the day-18 overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day52-cell.sh ladder <lockfd>
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
ladder)
    arms=(off base i6d i6g i9g fill i1 i2 i8 i5 i7 i4)
    bins=()
    for b in run-gen-final run-gen run-gen-i6 run-gen-i9 run-gen-fill run-gen-i1 run-gen-i2 run-gen-i8 run-gen-i5 run-gen-i7 run-gen-i4; do
        [ -x "$D40_BINS/$b" ] || { echo "missing binary $b"; exit 2; }
        bins+=("$D40_BINS/$b")
    done
    sha256sum "${bins[@]}" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    arm() { # $1 arm  $2 label
        local bin door=(--experts-via-tier --expert-bank-stages --expert-bank-host-bytes=17179869184)
        case $1 in
            off) bin=$D40_BINS/run-gen-final; door=() ;;
            base) bin=$D40_BINS/run-gen; door=(--experts-via-tier --expert-bank-stages) ;;
            i6d) bin=$D40_BINS/run-gen-i6; door=(--experts-via-tier --expert-bank-stages) ;;
            i6g) bin=$D40_BINS/run-gen-i6 ;;
            *) bin=$D40_BINS/run-gen-$1 ;;
        esac
        run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$bin" "$D40_ART" 55 88 13 "${door[@]}"
    }
    for i in 1 2 3 4 5; do for a in "${arms[@]}"; do arm "$a" "o1-$a-r$i"; done; done
    for i in 1 2 3 4 5; do for ((k = ${#arms[@]} - 1; k >= 0; k--)); do a=${arms[$k]}; arm "$a" "o2-$a-r$i"; done; done
    echo "ladder cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
