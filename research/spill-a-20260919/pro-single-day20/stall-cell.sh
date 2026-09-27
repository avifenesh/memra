#!/usr/bin/env bash
# memra#536 Move 2 cell (i), the capture arm, WP-A day 20 (DAY20.md pre-registration): the day-16 script's boot
# with the prefix cache ON at 1024 MB (a 4096-token entry is about 278 MB on the 27B) and the host tier at 8 GiB
# (the door needs it), door OFF or ON; the harness's `capture` mode (stall_cell.py: the prime arm's intruder
# plus an untimed same-prompt re-post that reads cached_tokens). MEMRA_SERVE_SPEC=0 so the tenant emits one
# token per tick. One collector lock hold per BOOT. usage: stall-cell.sh <lockfd> off|on
set -uo pipefail
fd=$1
which=$2
R=/root/spill-receipts/a-day20
cell=stall-capture-$which
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
  off|on)
    extra="MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192"
    [ "$which" = on ] && extra="$extra MEMRA_KV_HOST_CONTRACTS=1"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
    mark boot; boot "$extra" "$EV/server.log" || exit 1; mark ready
    python3 $H --port "$PORT" --mode capture --server-log "$EV/server.log" --out "$EV/capture" --tag "stall-capture-$which" | tee "$EV/capture.log" || rc=1
    mark capture-done
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1 ;;
  *) echo "usage: stall-cell.sh <lockfd> off|on"; exit 2 ;;
esac
stop; mark stopped
echo "$cell rc=$rc"
exit $rc
