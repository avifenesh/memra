#!/usr/bin/env bash
# memra#536 stall cell driver, day 17 demote arm (DAY17.md pre-registration; the day-16 script with the receipts root moved). One collector lock
# hold per BOOT: `prime` (cache off: the prime alone), `off` (cache 256 MB + host tier 8 GiB, door OFF: the
# demote and promote arms), `on` (the same with MEMRA_KV_HOST_CONTRACTS=1). Every boot MEMRA_SERVE_SPEC=0
# so the tenant emits one token per tick. The harness is stall_cell.py (client side, stdlib).
# usage: stall-cell.sh <lockfd> prime|off|on
set -uo pipefail
fd=$1
which=$2
R=/root/spill-receipts/a-day17
cell=stall-$which
EV=$R/$cell/ev
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum $R/bins/memra-server | tee "$EV/binary.sha256"
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/memra-server
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
H=research/spill-a-20260919/stall_cell.py
: > "$EV/marks.tsv"
rc=0
case $which in
  prime)
    mark boot; boot "MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0" "$EV/server.log" || exit 1; mark ready
    python3 $H --port "$PORT" --mode prime --server-log "$EV/server.log" --out "$EV/prime" --tag stall-prime | tee "$EV/prime.log" || rc=1
    mark prime-done ;;
  off|on)
    extra="MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192"
    [ "$which" = on ] && extra="$extra MEMRA_KV_HOST_CONTRACTS=1"
    mark boot; boot "$extra" "$EV/server.log" || exit 1; mark ready
    python3 $H --port "$PORT" --mode demote --server-log "$EV/server.log" --out "$EV/demote" --tag "stall-demote-$which" | tee "$EV/demote.log" || rc=1
    mark demote-done
    python3 $H --port "$PORT" --mode promote --server-log "$EV/server.log" --out "$EV/promote" --tag "stall-promote-$which" | tee "$EV/promote.log" || rc=1
    mark promote-done ;;
  *) echo "usage: stall-cell.sh <lockfd> prime|off|on"; exit 2 ;;
esac
stop; mark stopped
echo "$cell rc=$rc"
exit $rc
