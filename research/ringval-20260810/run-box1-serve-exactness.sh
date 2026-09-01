#!/usr/bin/env bash
# Fresh-boot b1fix golden and c=4 burst receipts with the SWA ring enabled.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target-ringval}
BIN=${BIN:-$TARGET/release/memra-server}
QOS=${QOS:-$REPO/research/p0iso-20260810/qos_probe.py}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-stepfun/step-3.7-flash}
GOLDEN=${GOLDEN:-$HOME/darktrain2/golden-response.bin}
PORT=${PORT:-18432}
BASE=http://127.0.0.1:$PORT
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/serve-exactness-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_BINARY=${EXPECTED_BINARY:-7f04f76715d637c46a379366a833d518aed9d465a5dcfd1ffee53be79d9b9cef}
EXPECTED_GOLDEN=${EXPECTED_GOLDEN:-21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de}
EXPECTED_QOS=${EXPECTED_QOS:-6c9e7386e3304deb6b625db1e7bd5089b3f0cf4844c198b17d7173e5c0082e9d}
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
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$log"; return 1; }
        sleep 1
    done
    tail -100 "$log"
    return 1
}

preflight() {
    local source binary golden qos apps
    source=$(git -C "$REPO" rev-parse HEAD)
    binary=$(sha256sum "$BIN" | awk '{print $1}')
    golden=$(sha256sum "$GOLDEN" | awk '{print $1}')
    qos=$(sha256sum "$QOS" | awk '{print $1}')
    echo "source_commit=$source"
    echo "binary_sha256=$binary"
    echo "golden_sha256=$golden"
    echo "qos_sha256=$qos"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT" "$GOLDEN"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $binary == "$EXPECTED_BINARY" ]]
    [[ $golden == "$EXPECTED_GOLDEN" ]]
    [[ $qos == "$EXPECTED_QOS" ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; return 1; }
}

start_server() {
    local cell=$1 log=$1/server.log
    {
        echo 'MEMRA_SWA_RING=1'
        echo 'MEMRA_PP_STAGES=2'
        echo 'MEMRA_PP_DEVICES=0,1'
        echo 'MEMRA_CTX=262144'
        echo 'MEMRA_MOE_GROUPED=1'
        echo 'MEMRA_PREFILL_TICK=2048'
        echo 'MEMRA_PREFIX_CACHE_MB=256'
        echo 'MEMRA_PREFIX_DEDUP=1'
        echo 'MEMRA_PRIME_BATCH_HOLD_MS=4'
    } >"$cell/server-env.txt"
    env -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_BG_JOB \
        -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_SWA_RING=1 \
        MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PREFIX_CACHE_MB=256 \
        MEMRA_PREFIX_DEDUP=1 \
        MEMRA_PRIME_BATCH_HOLD_MS=4 \
        MEMRA_TICK_TRACE=1 \
        "$BIN" >"$log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$log"
    grep -q '\[admission\].*capped at 4639 rows' "$log"
}

run_cell() {
    local label=$1 requests=$2 cell=$OUT/$1 rc
    mkdir -p "$cell"
    echo "cell=$label requests=$requests boot_start=$(date -u +%FT%TZ)"
    start_server "$cell"
    curl -sf "$BASE/metrics" >"$cell/metrics-before.json"
    snapshot "$cell/nvidia-smi-ready.log" ready
    nvidia-smi \
        --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$cell/gpu.csv" 2>&1 &
    SAMPLER_PID=$!
    set +e
    "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "$label" \
        --requests "$requests" --max-tokens 64 --golden "$GOLDEN" \
        --rows "$cell/qos-rows.jsonl" --summary "$cell/qos-summary.json"
    rc=$?
    set -e
    echo "$rc" >"$cell/probe-exit-code.txt"
    stop_sampler
    curl -sf "$BASE/metrics" >"$cell/metrics-after.json" || true
    snapshot "$cell/nvidia-smi-after-probe.log" after-probe
    stop_server
    [[ $rc -eq 0 ]]
    python3 - "$cell/qos-summary.json" "$requests" "$EXPECTED_GOLDEN" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
want_n = int(sys.argv[2])
want_hash = sys.argv[3]
assert summary["n_ok"] == want_n
assert summary["n_error"] == 0
assert summary["golden_matches"] == want_n
assert summary["golden_divergences"] == 0
assert summary["hash_counts"] == {want_hash: want_n}
assert summary["exactness"] == "match"
PY
    if grep -Ein 'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died' "$cell/server.log"; then
        return 1
    fi
    echo "cell=$label verdict=PASS done=$(date -u +%FT%TZ)"
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    preflight
    snapshot "$OUT/nvidia-smi-before.log" preflight
    run_cell golden-c1 1
    run_cell barrier-c4 4
    snapshot "$OUT/nvidia-smi-after.log" final
    echo 'one_hash_flag_on=PASS'
    echo 'c4_burst_flag_on=PASS'
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
