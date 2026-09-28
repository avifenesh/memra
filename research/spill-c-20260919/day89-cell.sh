#!/usr/bin/env bash
# Day 89 (DAY91.md section 2): day88-cell.sh with the promoted build labelled i24 (I24, 1fd4b24c0). The day-88 cells `promo`, `promo-res` and `promo-spec` (lane/spill-c-20260919, research/spill-c-20260919/DAY88.md
# section 5, registered before this script): the MoE slot cache door's promotion, phase 1.
# promo: the spill shape (MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986, prompt 55 88 13); arms naked
#   (run-gen-i24, no flag), q22 (run-gen-i22 with --experts-via-tier --expert-bank-host-bytes=17179869184), legacy
#   (i24, MEMRA_EXPERTS_VIA_TIER=0), alloc (i24, --expert-bank-pool-allocated), nopf (i24, MEMRA_MOE_PREFETCH=0),
#   legnopf (i24, MEMRA_EXPERTS_VIA_TIER=0 MEMRA_MOE_PREFETCH=0); order 1 x 5, order 2 reversed x 5.
# promo-res: the resident shape (the same argv without MEMRA_MOE_RESIDENT and MEMRA_MOE_SLOTS); arms naked and
#   legacy; order 1 x 5, order 2 reversed x 5.
# promo-spec: run-spec-i24 in the spill shape, K=1..8 (the binary's own sweep), by default and with
#   MEMRA_EXPERTS_VIA_TIER=0.
# One collector lock hold per cell. The day-18 overlap environment.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir, D40_TREE worktree, D40_ART the approved
# artifact, D40_LOCK the rig lock path.
# usage: day88-cell.sh promo|promo-res|promo-spec <lockfd>
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
promo|promo-res)
    # DAY88 section 6a: each cell hashes the binaries it runs (promo-res runs run-gen-i24 alone).
    if [ "$cell" = promo ]; then
        sha256sum "$D40_BINS/run-gen-i24" "$D40_BINS/run-gen-i22" | tee "$EV/binary.sha256"
    else
        sha256sum "$D40_BINS/run-gen-i24" | tee "$EV/binary.sha256"
    fi
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    shape=(MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=9986)
    arms=(naked q22 legacy alloc nopf legnopf)
    if [ "$cell" = promo-res ]; then shape=(); arms=(naked legacy); fi
    arm() { # $1 arm  $2 label
        local pre=() post=() bin=run-gen-i24
        case $1 in
            q22) bin=run-gen-i22; post=(--experts-via-tier --expert-bank-host-bytes=17179869184) ;;
            legacy) pre=(MEMRA_EXPERTS_VIA_TIER=0) ;;
            alloc) post=(--expert-bank-pool-allocated) ;;
            nopf) pre=(MEMRA_MOE_PREFETCH=0) ;;
            legnopf) pre=(MEMRA_EXPERTS_VIA_TIER=0 MEMRA_MOE_PREFETCH=0) ;;
        esac
        run_gen "$2" env MEMRA_NGEN=32 "${shape[@]}" "${pre[@]}" "$D40_BINS/$bin" "$D40_ART" 55 88 13 "${post[@]}"
    }
    reversed=(); for ((k = ${#arms[@]} - 1; k >= 0; k--)); do reversed+=("${arms[k]}"); done
    for i in 1 2 3 4 5; do for a in "${arms[@]}"; do arm "$a" "o1-$a-r$i"; done; done
    for i in 1 2 3 4 5; do for a in "${reversed[@]}"; do arm "$a" "o2-$a-r$i"; done; done
    echo "$cell cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
promo-spec)
    sha256sum "$D40_BINS/run-spec-i24" | tee "$EV/binary.sha256"
    run_gen spec-naked env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$D40_BINS/run-spec-i24" "$D40_ART" 55 88 13
    run_gen spec-legacy env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 MEMRA_EXPERTS_VIA_TIER=0 "$D40_BINS/run-spec-i24" "$D40_ART" 55 88 13
    echo "promo-spec cell done: naked rc=$(cat "$EV/spec-naked.exit") legacy rc=$(cat "$EV/spec-legacy.exit")"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
