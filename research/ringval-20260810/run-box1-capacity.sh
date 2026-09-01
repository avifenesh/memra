#!/usr/bin/env bash
# Honest c=24 full-262k first-defer receipt, fresh server for SWA ring OFF and ON.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target-ringval}
BIN=${BIN:-$TARGET/release/memra-server}
WORKLOAD=${WORKLOAD:-$REPO/research/capbase-20260809/run_workloads.py}
ANALYZER=${ANALYZER:-$REPO/research/capbase-20260809/analyze_capacity.py}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PORT=${PORT:-18433}
BASE=http://127.0.0.1:$PORT
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/capacity-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_BINARY=${EXPECTED_BINARY:-7f04f76715d637c46a379366a833d518aed9d465a5dcfd1ffee53be79d9b9cef}
SERVER_PID=0
SAMPLER_PID=0

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw,power.limit \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

stop_sampler() {
    if (( SAMPLER_PID > 0 )); then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=0
    fi
}

stop_server() {
    stop_sampler
    if (( SERVER_PID > 0 )); then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=0
    fi
    for _ in $(seq 1 180); do
        [[ -z $(compute_apps) ]] && return 0
        sleep 1
    done
    compute_apps
    return 1
}

cleanup() {
    stop_server || true
}
trap cleanup EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -120 "$log"; return 1; }
        sleep 1
    done
    tail -120 "$log"
    return 1
}

preflight() {
    local source binary apps
    source=$(git -C "$REPO" rev-parse HEAD)
    binary=$(sha256sum "$BIN" | awk '{print $1}')
    echo "source_commit=$source"
    echo "binary_sha256=$binary"
    echo "workload_sha256=$(sha256sum "$WORKLOAD" | awk '{print $1}')"
    echo "analyzer_sha256=$(sha256sum "$ANALYZER" | awk '{print $1}')"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $binary == "$EXPECTED_BINARY" ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; return 1; }
}

start_server() {
    local arm=$1 cell=$2 log=$2/server.log
    local -a ring_env=()
    [[ $arm == on ]] && ring_env+=(MEMRA_SWA_RING=1)
    {
        echo "MEMRA_SWA_RING=$([[ $arm == on ]] && echo 1 || echo unset)"
        echo 'MEMRA_PP_STAGES=2'
        echo 'MEMRA_PP_DEVICES=0,1'
        echo 'MEMRA_CTX=262144'
        echo 'MEMRA_MOE_GROUPED=1'
        echo 'MEMRA_PREFILL_TICK=2048'
        echo 'MEMRA_MAX_SESSIONS=64'
        echo 'MEMRA_REUSE_POOL=2'
        echo 'MEMRA_PREFIX_CACHE_MB=0'
    } >"$cell/server-env.txt"
    env -u MEMRA_SWA_RING -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH \
        -u MEMRA_API_KEY \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="step=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_CTX=262144 \
        MEMRA_MAX_SESSIONS=64 \
        MEMRA_REUSE_POOL=2 \
        MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        "${ring_env[@]}" \
        "$BIN" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$log"
    if [[ $arm == on ]]; then
        grep -q '\[admission\].*capped at 4639 rows' "$log"
    else
        if grep -q '\[admission\].*capped at' "$log"; then
            return 1
        fi
    fi
}

run_arm() {
    local arm=$1 cell=$OUT/$1 client_rc analyzer_rc
    mkdir -p "$cell"
    echo "arm=$arm n=1 concurrency=24 start=$(date -u +%FT%TZ)"
    start_server "$arm" "$cell"
    curl -sf "$BASE/metrics" >"$cell/metrics-before.json"
    snapshot "$cell/nvidia-smi-ready.log" ready
    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free \
        --format=csv,noheader,nounits -l 1 >"$cell/gpu.csv" 2>&1 &
    SAMPLER_PID=$!
    set +e
    timeout 43200 python3 "$WORKLOAD" capacity \
        "$BASE" "$cell/requests.jsonl" --max-ctx 262144 --concurrency 24 --max-tokens 64 \
        2>&1 | tee "$cell/client.log"
    client_rc=${PIPESTATUS[0]}
    set -e
    echo "$client_rc" >"$cell/client-exit-code.txt"
    curl -sf "$BASE/metrics" >"$cell/metrics-final.json" || true
    snapshot "$cell/nvidia-smi-before-stop.log" before-stop
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
        >"$cell/compute-apps-final.txt" 2>&1 || true
    stop_server
    grep -n -E '\[admission\]|reclaim-on-defer|VRAM defer|\[spec-k\]' "$cell/server.log" \
        >"$cell/admission-lines.txt" || true
    grep -n -E 'CUDA_ERROR|out of memory|panicked at|memory allocation.*failed' "$cell/server.log" \
        >"$cell/failure-lines.txt" || true
    set +e
    python3 "$ANALYZER" "$cell" --out "$cell/capacity-summary.json" \
        2>&1 | tee "$cell/capacity-summary.log"
    analyzer_rc=${PIPESTATUS[0]}
    set -e
    echo "$analyzer_rc" >"$cell/analyzer-exit-code.txt"
    [[ $client_rc -eq 0 && $analyzer_rc -eq 0 ]]
    [[ ! -s $cell/failure-lines.txt ]]
    python3 - "$cell/capacity-summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["n"] == 1
assert summary["requested_max_ctx"] == 262144
assert summary["offered_concurrency"] == 24
assert summary["requests_ok"] == summary["requests_n"] == 24
assert summary["capacity_result"]["kind"] == "first_defer"
assert summary["first_defer"] is not None
assert not summary["captured_failure_lines"]
assert summary["step_oom_parks"] == 0
PY
    echo "arm=$arm verdict=PASS done=$(date -u +%FT%TZ)"
}

reduce() {
    python3 - "$OUT/off/capacity-summary.json" "$OUT/on/capacity-summary.json" \
        "$OUT/comparison.json" <<'PY'
import json
import sys

off = json.load(open(sys.argv[1], encoding="utf-8"))
on = json.load(open(sys.argv[2], encoding="utf-8"))
off_n = off["capacity_result"]["sessions"]
on_n = on["capacity_result"]["sessions"]
result = {
    "n_per_arm": 1,
    "offered_concurrency": 24,
    "requested_max_ctx": 262144,
    "ring_off_sessions_before_first_defer": off_n,
    "ring_on_sessions_before_first_defer": on_n,
    "measured_session_capacity_ratio": on_n / off_n,
    "requests_ok": {"off": off["requests_ok"], "on": on["requests_ok"]},
    "step_oom_parks": {"off": off["step_oom_parks"], "on": on["step_oom_parks"]},
}
with open(sys.argv[3], "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write("\n")
print(json.dumps(result, sort_keys=True))
PY
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    preflight
    snapshot "$OUT/nvidia-smi-before.log" preflight
    run_arm off
    run_arm on
    reduce
    snapshot "$OUT/nvidia-smi-after.log" final
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
