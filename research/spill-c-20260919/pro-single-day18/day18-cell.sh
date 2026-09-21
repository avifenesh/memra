#!/usr/bin/env bash
# Day 18 cells (lane/spill-c-20260919, research/spill-c-20260919/DAY18.md, pre-registered before any run).
# One collector lock hold per cell (tools/tier-battery.py --external-lock passes the lock fd as $2).
# Cells: hashlock (MoeSlotCache door item 3, hash lock red arm + control), overlap (item 4 timing pair,
# N=5 per arm per order, both orders), serverdoor (item 6 red arm: memra-server with the door flag),
# hashmicro (HOSTPREFIX door review input: engine checksum over cached / WC pinned / heap bytes).
# Environment (set by the driver, never hosts or ids): D18_R receipts root, D18_BINS binary dir,
# D18_TREE worktree, D18_ART artifact for the door arms, D18_ART_OTHER a non-approved artifact,
# D18_LOCK the rig lock path, D18_PORT server port, D18_MOE_ENV extra env for the model (e.g. resident).
# usage: day18-cell.sh <cell> <lockfd>
set -uo pipefail
cell=$1; fd=$2
: "${D18_R:?}" "${D18_BINS:?}" "${D18_TREE:?}" "${D18_LOCK:?}"
EV=$D18_R/$cell/ev
mkdir -p "$EV"
cd "$D18_TREE" || exit 1
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$D18_LOCK" --owner collector > "$EV/LOCK.json"
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
run_gen() { # $1 label  $2.. argv (env words first)
    local label=$1; shift
    mark "$label start"
    "$@" 2>&1 | stamp "$EV/$label.log"
    local rc=${PIPESTATUS[0]}
    echo "$rc" > "$EV/$label.exit"
    mark "$label end rc=$rc"
    return 0
}
case $cell in
hashlock)
    : "${D18_ART_OTHER:?}"
    sha256sum "$D18_BINS/run-gen" | tee "$EV/binary.sha256"
    sha256sum "$D18_ART_OTHER" | tee "$EV/artifact.sha256"
    # shellcheck disable=SC2086
    run_gen door    env $D18_MOE_ENV MEMRA_NGEN=8 "$D18_BINS/run-gen" "$D18_ART_OTHER" 55 88 13 --experts-via-tier
    # shellcheck disable=SC2086
    run_gen control env $D18_MOE_ENV MEMRA_NGEN=8 "$D18_BINS/run-gen" "$D18_ART_OTHER" 55 88 13
    echo "hashlock cell done: door rc=$(cat "$EV/door.exit") control rc=$(cat "$EV/control.exit")"
    ;;
overlap)
    : "${D18_ART:?}"
    sha256sum "$D18_BINS/run-gen" | tee "$EV/binary.sha256"
    arm() { # $1 off|on  $2 label
        if [ "$1" = on ]; then
            run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$D18_BINS/run-gen" "$D18_ART" 55 88 13 --experts-via-tier
        else
            run_gen "$2" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$D18_BINS/run-gen" "$D18_ART" 55 88 13
        fi
    }
    for i in 1 2 3 4 5; do arm off "o1-off-r$i"; arm on "o1-on-r$i"; done
    for i in 1 2 3 4 5; do arm on "o2-on-r$i"; arm off "o2-off-r$i"; done
    echo "overlap cell done: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    ;;
serverdoor)
    : "${D18_ART:?}" "${D18_PORT:?}"
    sha256sum "$D18_BINS/memra-server" | tee "$EV/binary.sha256"
    . tools/port-guard.sh
    memra_port_guard day18-serverdoor "$D18_PORT" D18_PORT || exit 1
    if curl -s --max-time 1 "http://127.0.0.1:$D18_PORT/v1/models" >/dev/null 2>&1; then
        echo "port $D18_PORT already serving, refusing to boot over it"; exit 1
    fi
    log=$EV/server.log
    mark "boot"
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$D18_ART" "MEMRA_ADDR=127.0.0.1:$D18_PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 $D18_MOE_ENV "$D18_BINS/memra-server" --experts-via-tier >"$log" 2>&1 &
    SERVER_PID=$!
    ready=0
    for _ in $(seq 1 240); do
        curl -s --max-time 2 "http://127.0.0.1:$D18_PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 2
    done
    echo "ready=$ready" | tee "$EV/ready.txt"
    mark "ready=$ready"
    if [ "$ready" = 1 ]; then
        python3 - "$D18_PORT" "$EV/r1.json" <<'PYEOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
body = {"model": "gate", "prompt": "Name three primary colors, comma-separated.", "max_tokens": 16, "temperature": 0}
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
json.dump(json.load(r), open(out, "w"), indent=1)
PYEOF
        echo "request rc=$?" | tee "$EV/request.txt"
        mark "r1 done"
    fi
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null; echo "server exit=$?" | tee "$EV/server.exit"
    mark "stopped"
    echo "serverdoor cell done: door lines=$(grep -c 'experts-via-tier\|expert-host-slru' "$log")"
    ;;
hashmicro)
    sha256sum "$D18_BINS/hash-micro" | tee "$EV/binary.sha256"
    run_gen hash-micro "$D18_BINS/hash-micro" --bytes 167772160 --n 5
    grep 'HASH-MICRO rule' "$EV/hash-micro.log" || { echo "no rule line"; exit 1; }
    ;;
*) echo "unknown cell $cell"; exit 2;;
esac
