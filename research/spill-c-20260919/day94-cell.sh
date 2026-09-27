#!/usr/bin/env bash
# Day 94 cell `where285` (lane/spill-c-20260919, research/spill-c-20260919/DAY94.md section 1, registered before this
# script): where the 285K class's gen-only gap sits. Binaries run-gen-i25 (the naked door, untraced) and run-gen-i24
# (the door as the I24 sitting ran it, traced). The spill shape of DAY88 section 5 (MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32
# MEMRA_MOE_SLOTS=9986, prompt 55 88 13). Arms, each run under its own affinity (taskset -c):
#   wn   i25 naked, the box cap's cores (D94_WIDE)        wl   i25 legacy (MEMRA_EXPERTS_VIA_TIER=0), the same cores
#   pn   i25 naked, the P-cores alone (D94_PCORES)        pl   i25 legacy, the P-cores alone
#   wnc  wn with --moe-dispatch-clock                     wlc  wl with --moe-dispatch-clock
#   pnc  pn with --moe-dispatch-clock                     plc  pl with --moe-dispatch-clock
#   wo   i24 naked, the box cap's cores (the I24 sitting's program, beside)
# Order 1 (wn wl pn pl wnc wlc pnc plc wo) x 5, order 2 reversed x 5: 90 timed runs. Then two untimed traced twins of
# pn (t-pnt-r1, t-pnt-r2: --expert-bank-trace) for the host demand sequence. One collector lock hold.
# Environment (set by the driver): D40_R, D40_BINS, D40_TREE, D40_ART, D40_LOCK; D94_WIDE and D94_PCORES, the two CPU
# lists (the driver refuses to start without a P-core list distinct from the wide one).
# usage: day94-cell.sh where285 <lockfd>
set -uo pipefail
cell=$1; fd=$2
: "${D40_R:?}" "${D40_BINS:?}" "${D40_TREE:?}" "${D40_LOCK:?}" "${D40_ART:?}" "${D94_WIDE:?}" "${D94_PCORES:?}"
[ "$cell" = where285 ] || { echo "unknown cell $cell"; exit 2; }
EV=$D40_R/$cell/ev
mkdir -p "$EV"
cd "$D40_TREE" || exit 1
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$D40_LOCK" --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/tree.sha"
printf 'wide=%s\npcores=%s\n' "$D94_WIDE" "$D94_PCORES" | tee "$EV/cpus.txt"
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
: > "$EV/marks.tsv"
stamp() { python3 -c '
import sys, datetime
for line in sys.stdin.buffer:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    sys.stdout.buffer.write(ts.encode() + b"\t" + line); sys.stdout.buffer.flush()
' > "$1"; }
snap() {
    { date -u +%FT%T.%3NZ; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
      grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$EV/$1.snap"
}
run_gen() { # $1 label  $2 cpu list  $3.. argv (env words first)
    local label=$1 cpus=$2; shift 2
    snap "$label.before"
    mark "$label start"
    taskset -c "$cpus" "$@" 2>&1 | stamp "$EV/$label.log"
    local rc=${PIPESTATUS[0]}
    echo "$rc" > "$EV/$label.exit"
    echo "$cpus" > "$EV/$label.cpus"
    mark "$label end rc=$rc"
    snap "$label.after"
    return 0
}
sha256sum "$D40_BINS/run-gen-i25" "$D40_BINS/run-gen-i24" | tee "$EV/binary.sha256"
stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
[ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
shape=(MEMRA_NGEN=32 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=9986)
arm() { # $1 arm  $2 label
    local pre=() post=() bin=run-gen-i25 cpus=$D94_WIDE
    case $1 in
        p*) cpus=$D94_PCORES ;;
    esac
    case $1 in
        ?l|?lc) pre=(MEMRA_EXPERTS_VIA_TIER=0) ;;
    esac
    case $1 in
        *c) post=(--moe-dispatch-clock) ;;
    esac
    [ "$1" = wo ] && bin=run-gen-i24
    run_gen "$2" "$cpus" env "${shape[@]}" "${pre[@]}" "$D40_BINS/$bin" "$D40_ART" 55 88 13 "${post[@]}"
}
arms=(wn wl pn pl wnc wlc pnc plc wo)
reversed=(); for ((k = ${#arms[@]} - 1; k >= 0; k--)); do reversed+=("${arms[k]}"); done
for i in 1 2 3 4 5; do for a in "${arms[@]}"; do arm "$a" "o1-$a-r$i"; done; done
for i in 1 2 3 4 5; do for a in "${reversed[@]}"; do arm "$a" "o2-$a-r$i"; done; done
for i in 1 2; do
    run_gen "t-pnt-r$i" "$D94_PCORES" env "${shape[@]}" "$D40_BINS/run-gen-i25" "$D40_ART" 55 88 13 --expert-bank-trace
done
echo "$cell cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
