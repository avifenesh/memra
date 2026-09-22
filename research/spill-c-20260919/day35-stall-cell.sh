#!/usr/bin/env bash
# Day 35: the tenant-stall cell on the local RTX 5090 Laptop GPU with the 9B artifact (HOSTPREFIX-DOOR.md section D
# item 5's tail "no tenant-stall cell exists on this class"; DOOR-DECISION-PACKET.md item 7). Lane A's day-16 shape
# (pro-single-day16/stall-cell.sh, harness stall_cell.py) adapted only in what this card and model need, pre-registered
# in DAY35.md before this ran: the 9B, the day-31 budgets (MEMRA_PREFIX_CACHE_MB=64: one 54.6 MB plain 64-token entry
# fits the 67,108,864 B budget, two do not; MEMRA_KV_HOST_MB=8192), the day-16 boot (MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4
# MEMRA_SERVE_SPEC=0: one token per tick), the tenant's window lengthened to 400 tokens (day35_stall_cell.py
# --tenant-max-tokens 400; A's intruder shapes byte-for-byte). Six boots in ONE collector lock hold on
# /tmp/memra-5090.lock, two passes in opposite order (the day-28 form):
#   pass 1: prime (cache off: MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0), off (door OFF), on (MEMRA_KV_HOST_CONTRACTS=1)
#   pass 2: on, off, prime
# The prime boot runs --mode prime; the off and on boots run --mode demote then --mode promote (A's day-16 five arms:
# prime, demote OFF, demote ON, promote OFF, promote ON), N=5 per arm per order, both orders inside every harness run.
# The collector samples the card at 250 ms; this script's own 1 s sampler writes ev/card.during.csv beside it. Every
# receipt is replayed (`STALL REPLAY`). Every cell is executed-not-qualified development evidence. No host, id or
# price here; nothing here is compared to the target card.
# usage: day35-stall-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <server_bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5
cd "$TREE" || exit 1
EV=$R/stall/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN" | tee "$EV/binary.sha256"
H=research/spill-c-20260919/day35_stall_cell.py
A=research/spill-a-20260919/stall_cell.py
diff -u "$A" "$H" > "$EV/harness.diff"
TENANT_TOKENS=400
CACHE_MB=64
HOST_MB=8192
PORT=${MEMRA_GATE_PORT:-18131}
{
    echo "tree=$(git rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum "$H" | cut -d' ' -f1)"
    echo "a_harness_sha256=$(sha256sum "$A" | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "cache_mb=$CACHE_MB host_mb=$HOST_MB port=$PORT tenant_max_tokens=$TENANT_TOKENS fire_at=24"
    echo "passes=2 boots_per_pass=3 n_per_arm_per_order=5 pass1=prime,off,on pass2=on,off,prime"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
snap() { # label
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.$1.csv" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw,pstate --format=csv > "$EV/card.$1.csv" 2>&1
}
snap before
# shellcheck disable=SC1091
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard day35-stall-cell "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"; return 1
    fi
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 \
        $1 "$BIN" >"$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died during boot:"; tail -20 "$2"; return 1; }
        sleep 2
    done
    echo "server never became ready"; return 1
}
stop() {
    [[ -n $SERVER_PID ]] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
SAMPLER=""
stop_sampler() {
    [[ -n $SAMPLER ]] || return 0
    kill -TERM "$SAMPLER" 2>/dev/null || true
    wait "$SAMPLER" 2>/dev/null || true
    SAMPLER=""
}
# shellcheck disable=SC2329
cleanup() { stop; stop_sampler; }
trap cleanup EXIT
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
card() { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used,pstate --format=csv > "$1" 2>&1; }
: > "$EV/marks.tsv"
: > "$EV/replays.log"
rc=0
# one boot: $1 pass dir  $2 kind (prime|off|on)
one_boot() {
    local D=$1/$2 kind=$2 extra modes m
    mkdir -p "$D"
    case $kind in
        prime) extra="MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0"; modes="prime" ;;
        off)   extra="MEMRA_PREFIX_CACHE_MB=$CACHE_MB MEMRA_KV_HOST_MB=$HOST_MB"; modes="demote promote" ;;
        on)    extra="MEMRA_PREFIX_CACHE_MB=$CACHE_MB MEMRA_KV_HOST_MB=$HOST_MB MEMRA_KV_HOST_CONTRACTS=1"; modes="demote promote" ;;
        *) echo "unknown boot kind $kind"; return 1 ;;
    esac
    echo "kind=$kind extra=$extra modes=$modes tenant_max_tokens=$TENANT_TOKENS" > "$D/BOOT.txt"
    card "$D/card.before.csv"
    mark "boot-$(basename "$1")-$kind"
    if ! boot "$extra" "$D/server.log"; then
        echo "boot $(basename "$1")/$kind FAILED" | tee -a "$EV/replays.log"; echo "boot-failed" >> "$D/BOOT.txt"; return 1
    fi
    mark "ready-$(basename "$1")-$kind"
    for m in $modes; do
        python3 "$H" --port "$PORT" --mode "$m" --n 5 --tenant-max-tokens "$TENANT_TOKENS" \
            --server-log "$D/server.log" --out "$D/$m" --tag "stall-$m-$kind" | tee "$D/$m.log" || rc=1
        mark "$m-done-$(basename "$1")-$kind"
    done
    stop; mark "stopped-$(basename "$1")-$kind"
    card "$D/card.after.csv"
    for m in $modes; do
        { echo "== $(basename "$1")/$kind/$m"; python3 "$H" --replay "$D/$m/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    done
    grep -E 'off the tick|request parked|receipt:|refused|latched|DISABLED|failed|not routed' "$D/server.log" > "$D/door-lines.txt" || true
    return 0
}
# This script's own 1 s sampler across the six boots (the collector samples at 250 ms beside it).
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu \
    --format=csv -lms 1000 > "$EV/card.during.csv" 2> "$EV/card.during.err" &
SAMPLER=$!
for kind in prime off on; do one_boot "$EV/pass1" "$kind" || rc=1; done
for kind in on off prime; do one_boot "$EV/pass2" "$kind" || rc=1; done
stop_sampler
snap after
echo "stall-day35 rc=$rc" | tee "$EV/exit.txt"
exit $rc
