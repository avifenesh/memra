#!/usr/bin/env bash
# cell.sh <receipt-dir> <binary> <cells-file> [VAR=value ...]
# Boots memra-server on both cards with the given env overrides (naked otherwise), waits for
# /readyz, runs each line of <cells-file> as bench.py arguments, snapshots /metrics, /healthz
# and /readyz between cells, and records 250 ms GPU telemetry. One scored campaign per box:
# holds /tmp/memra-gpu.lock for the whole boot.
set -euo pipefail
receipt=$1; bin=$2; cells=$3; shift 3
mkdir -p "$receipt"
exec 9>/tmp/memra-gpu.lock
flock -n 9 || { echo SLOT_BUSY; exit 75; }
server_pid=""; tele_pid=""
stop_all() {
    [[ -n "$tele_pid" ]] && kill "$tele_pid" 2>/dev/null || true
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 90); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" 2>/dev/null || true
    fi
}
trap 'rc=$?; stop_all; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$receipt/terminal-apps.csv"; echo "TERMINAL rc=$rc utc=$(date -u +%FT%TZ)" | tee -a "$receipt/controller.log"; exit $rc' EXIT
echo "START utc=$(date -u +%FT%TZ) bin=$bin env=$*" | tee "$receipt/controller.log"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$receipt/pre-apps.csv"
[[ ! -s "$receipt/pre-apps.csv" ]] || { echo "GPU_NOT_IDLE"; cat "$receipt/pre-apps.csv"; exit 76; }
sha256sum "$bin" > "$receipt/binary.sha256"
(cd "$(dirname "$bin")/../.." 2>/dev/null; true)
printf '%s\n' "$@" > "$receipt/env.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv -lms 250 > "$receipt/telemetry-250ms.csv" 2>&1 &
tele_pid=$!
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
KEY="dsv4f-cell-$RANDOM$RANDOM"
export PATH=/usr/local/cuda/bin:$PATH MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
env "$@" MEMRA_MODELS="dsv4f=${MODEL_DIR:-/data/dsv4f/nvfp4}" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY="$KEY" \
    setsid nohup "$bin" > "$receipt/serve.log" 2>&1 &
server_pid=$!
t0=$(date +%s); ready=0
for _ in $(seq 1 1800); do
    kill -0 "$server_pid" 2>/dev/null || break
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/readyz" || true)" == 200 ]] && { ready=1; break; }
    sleep 1
done
[[ "$ready" == 1 ]] || { echo "BOOT_FAIL"; tail -30 "$receipt/serve.log"; exit 71; }
echo "READY after=$(( $(date +%s) - t0 ))s port=$PORT" | tee -a "$receipt/controller.log"
snap() {
    for ep in metrics healthz readyz; do
        curl -s --max-time 10 -H "authorization: Bearer $KEY" "http://127.0.0.1:$PORT/$ep" > "$receipt/$1-$ep.txt" || true
    done
    curl -s --max-time 10 -H "authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" > "$receipt/$1-models.json" || true
}
snap boot
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$receipt/boot-vram.csv"
i=0
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    i=$((i+1))
    # shellcheck disable=SC2086
    python3 /root/box/bench.py --port "$PORT" --key "$KEY" --out "$receipt/cells.jsonl" $line 2>&1 | tee -a "$receipt/controller.log"
    snap "cell$i"
done < "$cells"
echo "CELLS_DONE utc=$(date -u +%FT%TZ)" | tee -a "$receipt/controller.log"
