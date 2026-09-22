#!/usr/bin/env bash
# Day 28 isolating stall cell (memra#536 Move 2 owed item 3; pre-registered in DAY28.md before this ran).
# Ten boots in ONE collector lock hold, two passes in opposite order, each boot's harness interleaving
# idle/arm in both orders with N=5 per arm per order (lane A's stall_cell.py program):
#   pass 1: prime, capture-off, capture-on, exact-off, exact-on
#   pass 2: exact-on, exact-off, capture-on, capture-off, prime
# prime        A's day-16 prime arm: prefix cache OFF (MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0), the intruder's
#              ~5120-token prime alone; the subtraction arm of the capture class.
# capture-off  A's day-20 capture arm with the device cache at 8192 MB (not A's 1024: at 1024 the ten fresh seeds
#              evict and DEMOTE inside the cell, 8 demote lines per boot in A's receipts, 300 ms each under the
#              door's on-tick hashes, and the stall metric is the single worst tick; at 8192 nothing evicts, so
#              the arm carries the capture class alone), host tier 8 GiB, door OFF; the same intruder prompt,
#              whose seed captures; the re-post's cached_tokens recorded.
# capture-on   the same with MEMRA_KV_HOST_CONTRACTS=1 (the capture on the copy stream).
# exact-off    the isolating restore arm (day28_stall_cell.py --mode restore-exact): an on-grid seed whose timed
#              re-post is a whole-entry zero-suffix hit, no prime of its own; door OFF.
# exact-on     the same with MEMRA_KV_HOST_CONTRACTS=1 (the rows on the copy stream, the request parked).
# Every boot MEMRA_SERVE_SPEC=0 (one token per tick), MEMRA_CTX=8192, MEMRA_MAX_SESSIONS=4, the 27B artifact.
# Harness: research/spill-c-20260919/day28_stall_cell.py (lane A's stall_cell.py plus the restore-exact arm; the
# diff is banked beside the receipts). Every cell is executed-not-qualified development evidence. No host, id
# or price here. usage: day28-stall-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <server_bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$TREE" || exit 1
EV=$R/stall/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN" | tee "$EV/binary.sha256"
H=research/spill-c-20260919/day28_stall_cell.py
A=research/spill-a-20260919/stall_cell.py
diff -u "$A" "$H" > "$EV/harness.diff"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum "$H" | cut -d' ' -f1)"
    echo "a_harness_sha256=$(sha256sum "$A" | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "passes=2 boots_per_pass=5 n_per_arm_per_order=5"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
PORT=${MEMRA_GATE_PORT:-18132}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard stall-cell "$PORT" MEMRA_GATE_PORT || return 1
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
trap stop EXIT
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
env_for() { # cell -> boot env
    case $1 in
        prime) echo "MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0" ;;
        *-off) echo "MEMRA_PREFIX_CACHE_MB=8192 MEMRA_KV_HOST_MB=8192" ;;
        *-on) echo "MEMRA_PREFIX_CACHE_MB=8192 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1" ;;
    esac
}
mode_for() { case $1 in prime) echo prime ;; capture-*) echo capture ;; exact-*) echo restore-exact ;; esac; }
: > "$EV/marks.tsv"
: > "$EV/replays.log"
rc=0
PASS1=(prime capture-off capture-on exact-off exact-on)
PASS2=(exact-on exact-off capture-on capture-off prime)
for pass in 1 2; do
    if [ "$pass" = 1 ]; then CELLS=("${PASS1[@]}"); else CELLS=("${PASS2[@]}"); fi
    for cell in "${CELLS[@]}"; do
        D=$EV/pass$pass/$cell; mkdir -p "$D"
        nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv,noheader > "$D/card.before.csv" 2>&1
        mark "boot-$pass-$cell"
        boot "$(env_for "$cell")" "$D/server.log" || { rc=1; echo "boot $pass/$cell failed" | tee -a "$EV/replays.log"; continue; }
        mark "ready-$pass-$cell"
        python3 "$H" --port "$PORT" --mode "$(mode_for "$cell")" --server-log "$D/server.log" --out "$D/run" --tag "stall-$cell" | tee "$D/harness.log"
        hrc=${PIPESTATUS[0]}; echo "$hrc" > "$D/harness.exit"; [ "$hrc" = 0 ] || rc=1
        mark "done-$pass-$cell"
        stop; mark "stopped-$pass-$cell"
        nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv,noheader > "$D/card.after.csv" 2>&1
        { echo "== pass$pass/$cell"; python3 "$H" --replay "$D/run/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    done
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
echo "stall-isolating rc=$rc" | tee "$EV/exit.txt"
exit $rc
