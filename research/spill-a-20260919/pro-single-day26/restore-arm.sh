#!/usr/bin/env bash
# WP-A day 26, acceptance clause (4) of proposal 1 (DAY25.md, ruling 36): the day-21 restore arm, door ON, on the tree
# that carries the promoted-pin refusal. The day-21 stall-cell.sh ON boot byte-for-byte (MEMRA_PREFIX_CACHE_MB=1024
# MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1, MEMRA_SERVE_SPEC=0, MEMRA_CTX=8192, MEMRA_MAX_SESSIONS=4), the
# harness's `restore` mode (one seeded prompt, re-posted as a whole-entry hit at the tenant's 24th token in every timed
# run; the entry was NOT promoted this admission, so the route must still park once and land), --n 50 so one boot
# carries 100 timed restore runs and 100 idle runs. One collector lock hold. Executed-not-qualified. No host, id or
# price here. usage: restore-arm.sh <lockfd> <tree> <receipts_root> <model.gguf> <bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$TREE" || exit 1
EV=$R/restore-arm/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN" | tee "$EV/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum research/spill-a-20260919/stall_cell.py | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "boots=one ON boot, --mode restore --n 50 (100 timed restore runs)"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
PORT=${MEMRA_GATE_PORT:-18132}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard restore-arm "$PORT" MEMRA_GATE_PORT || return 1
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
card() { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$1" 2>&1; }
H=research/spill-a-20260919/stall_cell.py
: > "$EV/marks.tsv"
rc=0
extra="MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1"
echo "arm=on env=$extra" > "$EV/BOOT.txt"
card "$EV/card.before.csv"
mark boot
if ! boot "$extra" "$EV/server.log"; then echo "boot FAILED" | tee "$EV/exit.txt"; exit 1; fi
mark ready
python3 $H --port "$PORT" --mode restore --n 50 --server-log "$EV/server.log" --out "$EV/restore" --tag "stall-restore-on" | tee "$EV/restore.log" || rc=1
mark restore-done
stop; mark stopped
card "$EV/card.after.csv"
{ echo "== restore-arm/restore arm=on"; python3 $H --replay "$EV/restore/receipt.json"; } | tee "$EV/replays.log" || rc=1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
echo "restore-arm rc=$rc" | tee "$EV/exit.txt"
exit $rc
