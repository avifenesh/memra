#!/usr/bin/env bash
# Day 39: day 37's two-binary tenant-stall cell (day37-stall-cell.sh) moved to the target card (one RTX PRO 6000
# Blackwell, collector rig `pro-single`, lock /tmp/memra-gpu.lock) and run with THREE binaries in ONE collector lock
# hold, interleaved in both orders (DAY39.md section 1, pre-registered and pushed before this ran). The binaries (built
# on the box by day39-box-build.sh from the SHAs of DAY39.md section 0): B0 `091a931c0` (day 37's base tree, no option
# (a)), B1 `9717e8d57` (option (a) without the D2H spans), B2 `160929a92` (option (a) plus A day 30's D2H spans). Six
# programs in this order:
#   p1-b0, p2-b1, p3-b2, p4-b2, p5-b1, p6-b0
# and each program is day 35's six boots unchanged:
#   pass 1: prime (cache off: MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0), off (door OFF), on (MEMRA_KV_HOST_CONTRACTS=1)
#   pass 2: on, off, prime
# The prime boot runs --mode prime; the off and on boots run --mode demote then --mode promote, N=5 per arm per
# order, both orders inside every harness run. The harness is day35_stall_cell.py byte-for-byte. What changed from
# day 37's script, all of it the rig: the lock (/tmp/memra-gpu.lock), the target card's budgets and tenant window
# (A day 16's shape: MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192, --tenant-max-tokens 160, A's constant), the port
# (18132, day 23's), the PATH export, three binaries and six programs, and each binary's tree read from the build
# receipt (the binaries are copies, not in their build tree). The collector samples the card at 250 ms; this
# script's own 1 s sampler writes ev/card.during.csv beside it. Every receipt is replayed (`STALL REPLAY`). Every
# cell is executed-not-qualified development evidence. No host, id or price here; nothing here is compared to the
# RTX 5090.
# usage: day39-stall-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <bin_dir>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BINDIR=$5
BIN=""
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
B0_BIN=$BINDIR/bin/B0-memra-server; B1_BIN=$BINDIR/bin/B1-memra-server; B2_BIN=$BINDIR/bin/B2-memra-server
cd "$TREE" || exit 1
EV=$R/stall/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
# each binary's tree from its build receipt; the copy must still hash to the recorded SHA-256
tree_of() { grep "^binary=$1-memra-server " "$BINDIR/builds.log" | sed -E 's/.* tree=([0-9a-f]+) .*/\1/'; }
sha_of() { grep "^binary=$1-memra-server " "$BINDIR/builds.log" | sed -E 's/.* sha256=([0-9a-f]+).*/\1/'; }
for b in B0 B1 B2; do
    got=$(sha256sum "$BINDIR/bin/$b-memra-server" | cut -d' ' -f1)
    [ -n "$(sha_of $b)" ] && [ "$got" = "$(sha_of $b)" ] || { echo "binary $b does not match its build receipt"; exit 1; }
done
{ for b in B0 B1 B2; do echo "$b tree=$(tree_of $b) $(sha256sum "$BINDIR/bin/$b-memra-server")"; done; } | tee "$EV/binary.sha256"
H=research/spill-c-20260919/day35_stall_cell.py
A=research/spill-a-20260919/stall_cell.py
diff -u "$A" "$H" > "$EV/harness.diff"
TENANT_TOKENS=160
CACHE_MB=256
HOST_MB=8192
PORT=${MEMRA_GATE_PORT:-18132}
{
    echo "tree=$(git rev-parse HEAD)"
    for b in B0 B1 B2; do echo "$b tree=$(tree_of $b) sha256=$(sha_of $b)"; done
    echo "harness_sha256=$(sha256sum "$H" | cut -d' ' -f1)"
    echo "a_harness_sha256=$(sha256sum "$A" | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "cache_mb=$CACHE_MB host_mb=$HOST_MB port=$PORT tenant_max_tokens=$TENANT_TOKENS fire_at=24"
    echo "programs=p1-b0,p2-b1,p3-b2,p4-b2,p5-b1,p6-b0"
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
    memra_port_guard day39-stall-cell "$PORT" MEMRA_GATE_PORT || return 1
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
# one boot: $1 pass dir  $2 kind (prime|off|on); $PROG and $BIN name the program and its binary
PROG=""
one_boot() {
    local D=$1/$2 kind=$2 extra modes m
    mkdir -p "$D"
    case $kind in
        prime) extra="MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0"; modes="prime" ;;
        off)   extra="MEMRA_PREFIX_CACHE_MB=$CACHE_MB MEMRA_KV_HOST_MB=$HOST_MB"; modes="demote promote" ;;
        on)    extra="MEMRA_PREFIX_CACHE_MB=$CACHE_MB MEMRA_KV_HOST_MB=$HOST_MB MEMRA_KV_HOST_CONTRACTS=1"; modes="demote promote" ;;
        *) echo "unknown boot kind $kind"; return 1 ;;
    esac
    echo "program=$PROG bin=$BIN kind=$kind extra=$extra modes=$modes tenant_max_tokens=$TENANT_TOKENS" > "$D/BOOT.txt"
    card "$D/card.before.csv"
    mark "boot-$PROG-$(basename "$1")-$kind"
    if ! boot "$extra" "$D/server.log"; then
        echo "boot $PROG/$(basename "$1")/$kind FAILED" | tee -a "$EV/replays.log"; echo "boot-failed" >> "$D/BOOT.txt"; return 1
    fi
    mark "ready-$PROG-$(basename "$1")-$kind"
    for m in $modes; do
        python3 "$H" --port "$PORT" --mode "$m" --n 5 --tenant-max-tokens "$TENANT_TOKENS" \
            --server-log "$D/server.log" --out "$D/$m" --tag "stall-$m-$kind" | tee "$D/$m.log" || rc=1
        mark "$m-done-$PROG-$(basename "$1")-$kind"
    done
    stop; mark "stopped-$PROG-$(basename "$1")-$kind"
    card "$D/card.after.csv"
    for m in $modes; do
        { echo "== $PROG/$(basename "$1")/$kind/$m"; python3 "$H" --replay "$D/$m/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    done
    grep -E 'off the tick|request parked|receipt:|refused|latched|DISABLED|failed|not routed' "$D/server.log" > "$D/door-lines.txt" || true
    return 0
}
# one program: $1 name  $2 binary; day 35's six boots in day 35's order
program() {
    PROG=$1; BIN=$2
    local P=$EV/$1 kind
    mkdir -p "$P"
    echo "program=$PROG bin=$BIN sha256=$(sha256sum "$BIN" | cut -d' ' -f1)" > "$P/PROGRAM.txt"
    for kind in prime off on; do one_boot "$P/pass1" "$kind" || rc=1; done
    for kind in on off prime; do one_boot "$P/pass2" "$kind" || rc=1; done
}
# This script's own 1 s sampler across the 36 boots (the collector samples at 250 ms beside it).
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu \
    --format=csv -lms 1000 > "$EV/card.during.csv" 2> "$EV/card.during.err" &
SAMPLER=$!
program p1-b0 "$B0_BIN"
program p2-b1 "$B1_BIN"
program p3-b2 "$B2_BIN"
program p4-b2 "$B2_BIN"
program p5-b1 "$B1_BIN"
program p6-b0 "$B0_BIN"
stop_sampler
snap after
echo "stall-day39 rc=$rc" | tee "$EV/exit.txt"
exit $rc
