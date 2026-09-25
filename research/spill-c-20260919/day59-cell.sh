#!/usr/bin/env bash
# Day 59 cells (lane/spill-c-20260919, research/spill-c-20260919/DAY59.md, pre-registered before any cell):
# MEMRA_MOE_PREFETCH=1's deciding cell on the legacy MoE slot cache. One collector lock hold per cell.
#   pfgates  G1 run-gen OFF and PF at the pressure (9,986 slots) and heavy (512 slots) shapes; G2 run-spec PF K=1..8.
#   pftime   the pressure shape, OFF and PF: order 1 (off, pf) x 5, order 2 (pf, off) x 5.
#   pfnaked  the naked shape (no MoE variable), the same interleave.
# Environment (set by the driver): D40_R receipts root, D40_BINS binary dir (run-gen-c60, run-spec-c60),
# D40_TREE worktree, D40_ART the approved artifact, D40_LOCK.
#   pfserve  G3 memra-server OFF then PF, the same seven greedy requests (three sequential, four concurrent).
# usage: day59-cell.sh <pfgates|pftime|pfnaked|pfserve> <lockfd>
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
pfgates)
    sha256sum "$D40_BINS/run-gen-c60" "$D40_BINS/run-spec-c60" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    for slots in 9986 512; do
        run_gen "tape-off-s$slots" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=$slots "$D40_BINS/run-gen-c60" "$D40_ART" 55 88 13
        run_gen "tape-pf-s$slots" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=$slots MEMRA_MOE_PREFETCH=1 "$D40_BINS/run-gen-c60" "$D40_ART" 55 88 13
    done
    run_gen spec-pf env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 MEMRA_MOE_PREFETCH=1 "$D40_BINS/run-spec-c60" "$D40_ART" 55 88 13
    echo "pfgates cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
pftime|pfnaked)
    sha256sum "$D40_BINS/run-gen-c60" | tee "$EV/binary.sha256"
    stat -c '%n %s %Y' "$D40_ART" | tee "$EV/artifact.stat"
    [ -f "$D40_ART.sha256" ] && cp "$D40_ART.sha256" "$EV/artifact.sha256"
    shape=(MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=9986)
    [ "$cell" = pfnaked ] && shape=()
    arm() { # $1 off|pf  $2 label
        local pre=()
        [ "$1" = pf ] && pre=(MEMRA_MOE_PREFETCH=1)
        run_gen "$2" env "${shape[@]}" MEMRA_NGEN=32 "${pre[@]}" "$D40_BINS/run-gen-c60" "$D40_ART" 55 88 13
    }
    for i in 1 2 3 4 5; do arm off "o1-off-r$i"; arm pf "o1-pf-r$i"; done
    for i in 1 2 3 4 5; do arm pf "o2-pf-r$i"; arm off "o2-off-r$i"; done
    echo "$cell cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
pfserve)
    # G3: memra-server booted OFF then PF on the pressure MoE environment, the same requests in each boot. Only the
    # server this cell started is ever stopped.
    sha256sum "$D40_BINS/memra-server-c60" | tee "$EV/binary.sha256"
    PORT=${D59_PORT:-18159}
    # shellcheck disable=SC1091
    . "$D40_TREE/tools/port-guard.sh"
    SERVER_PID=""
    stop() {
        [ -n "$SERVER_PID" ] || return 0
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    }
    trap stop EXIT
    for arm in off pf; do
        pre=()
        [ "$arm" = pf ] && pre=(MEMRA_MOE_PREFETCH=1)
        memra_port_guard day59-pfserve "$PORT" D59_PORT || exit 1
        mark "boot-$arm"
        env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$D40_ART" "MEMRA_ADDR=127.0.0.1:$PORT" \
            MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=9986 "${pre[@]}" \
            "$D40_BINS/memra-server-c60" > "$EV/serve-$arm-server.log" 2>&1 &
        SERVER_PID=$!
        ready=0
        for _ in $(seq 1 300); do
            curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && { ready=1; break; }
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 2
        done
        echo "$arm ready=$ready" | tee -a "$EV/serve.txt"
        [ "$ready" = 1 ] && python3 "$D40_TREE/research/spill-c-20260919/day59-serve-client.py" "$PORT" "$EV/serve-$arm"
        mark "requests-done-$arm"
        stop
        mark "stopped-$arm"
    done
    echo "pfserve cell done"
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
