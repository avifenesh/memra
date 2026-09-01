#!/usr/bin/env bash
# One-lock interleaved x5 Step-3.7 sigmoid-router A/B and CUDA transfer receipt.
set -euo pipefail

REPO=${SIGROUTER_REPO:-/home/ubuntu/memra-cx-sigrouter}
OUT=${SIGROUTER_PERF_OUT:-$REPO/research/sigrouter-20260811/raw/box1-perf}
MODEL_ROOT=${SIGROUTER_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${SIGROUTER_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
SERVER=$REPO/target/release/memra-server
RUN_GEN=$REPO/target/release/run-gen
BENCH=$REPO/research/newboxgates-20260811/serve_bench.py
REDUCE=$REPO/research/sigrouter-20260811/reduce-perf.py
EXTRACT_NSYS=$REPO/research/sigrouter-20260811/extract-nsys.py
PROMPT=$REPO/tools/fast-gate/prompts/probe.txt
PORT=${SIGROUTER_PERF_PORT:-18454}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=
sampler_pid=

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
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

wait_idle() {
    for _ in $(seq 1 120); do
        test -z "$(compute_apps)" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

stop_sampler() {
    local pid=${sampler_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sampler_pid=
}

stop_server() {
    stop_sampler
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            wait_idle
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"
    return 1
}
trap stop_server EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    tail -200 "$log"
    return 1
}

assert_server_clean() {
    local log=$1 failures
    failures=$(grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|illegal|sentinel' \
        "$log" || true)
    if [[ -n $failures ]]; then
        echo "$failures"
        echo "FAIL: server failure signature in $log"
        return 1
    fi
}

run_point() {
    local rep=$1 arm=$2 concurrency=$3
    local label="r${rep}-${arm}-c${concurrency}"
    echo "point=$label start=$(date -u +%FT%TZ)"
    snapshot "$OUT/${label}-thermal-before.log" "$label-before"
    python3 "$BENCH" \
        --base "$BASE" \
        --model "$MODEL_NAME" \
        --shape decode \
        --label "$label" \
        --concurrency "$concurrency" \
        --max-tokens 512 \
        --require-length \
        --out "$OUT/points.jsonl" \
        2>&1 | tee "$OUT/${label}-load.log"
    snapshot "$OUT/${label}-thermal-after.log" "$label-after"
    echo "point=$label done=$(date -u +%FT%TZ)"
}

run_arm() {
    local rep=$1 arm=$2 first_c=$3 second_c=$4
    local label="r${rep}-${arm}" log="$OUT/r${rep}-${arm}-server.log"
    local -a policy=()
    [[ $arm == rollback ]] && policy=(MEMRA_SIG_ROUTER=0)
    echo "arm=$label start=$(date -u +%FT%TZ)"
    snapshot "$OUT/${label}-thermal-before.log" "$label-before"
    {
        echo "arm=$arm"
        if [[ $arm == default ]]; then
            echo "MEMRA_SIG_ROUTER=<unset>"
        else
            echo "MEMRA_SIG_ROUTER=0"
        fi
        echo "MEMRA_SERVE_SPEC=0"
        echo "MEMRA_PP_STAGES=2"
        echo "MEMRA_PP_DEVICES=0,1"
        echo "MEMRA_PREFIX_CACHE_MB=0"
    } >"$OUT/${label}-env.txt"
    env \
        -u MEMRA_SIG_ROUTER \
        -u MEMRA_SERVE_BATCH \
        -u MEMRA_SPEC_K \
        -u MEMRA_BG_JOB \
        -u MEMRA_SERVE_B1FAST \
        "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="$MODEL_NAME=$MODEL" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_TAG="sigrouter-$label" \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
    python3 "$BENCH" \
        --base "$BASE" \
        --model "$MODEL_NAME" \
        --shape warmup \
        --label "$label-warmup" \
        --concurrency 1 \
        --max-tokens 16 \
        --out "$OUT/warmups.jsonl" \
        >"$OUT/${label}-warmup.log" 2>&1
    nvidia-smi \
        --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/${label}-gpu.csv" 2>&1 &
    sampler_pid=$!
    run_point "$rep" "$arm" "$first_c"
    run_point "$rep" "$arm" "$second_c"
    curl -sf "$BASE/metrics" >"$OUT/${label}-metrics.txt"
    stop_server
    assert_server_clean "$log"
    snapshot "$OUT/${label}-thermal-after.log" "$label-after"
    echo "arm=$label done=$(date -u +%FT%TZ)"
}

run_trace() {
    local arm=$1
    local -a policy=()
    [[ $arm == rollback ]] && policy=(MEMRA_SIG_ROUTER=0)
    local prefix="$OUT/nsys-$arm" repfile="$OUT/nsys-$arm.nsys-rep"
    echo "trace=$arm start=$(date -u +%FT%TZ)"
    env \
        -u MEMRA_SIG_ROUTER \
        "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_NGEN=1 \
        MEMRA_PROMPT_FILE="$PROMPT" \
        nsys profile \
            --force-overwrite=true \
            --trace=cuda,nvtx \
            --sample=none \
            --cpuctxsw=none \
            --output="$prefix" \
            "$RUN_GEN" "$MODEL" \
            2>&1 | tee "$OUT/nsys-$arm-run-gen.log"
    nsys stats --force-export=true --report cuda_api_sum --format csv "$repfile" \
        >"$OUT/nsys-$arm-cuda-api.csv" 2>"$OUT/nsys-$arm-cuda-api.stderr"
    nsys stats --report cuda_gpu_mem_size_sum --format csv "$repfile" \
        >"$OUT/nsys-$arm-mem-size.csv" 2>"$OUT/nsys-$arm-mem-size.stderr"
    nsys stats --report cuda_gpu_mem_time_sum --format csv "$repfile" \
        >"$OUT/nsys-$arm-mem-time.csv" 2>"$OUT/nsys-$arm-mem-time.stderr"
    nsys export --type=sqlite --force-overwrite=true --output="$OUT/nsys-$arm.sqlite" "$repfile" \
        >"$OUT/nsys-$arm-export.log" 2>&1
    python3 "$EXTRACT_NSYS" "$OUT/nsys-$arm.sqlite" "$OUT/nsys-$arm-memcpy.json" \
        >"$OUT/nsys-$arm-extract.log"
    sha256sum "$repfile" "$OUT/nsys-$arm.sqlite" >"$OUT/nsys-$arm-SHA256SUMS"
    echo "trace=$arm done=$(date -u +%FT%TZ)"
}

for artifact in "$MODEL" "$SERVER" "$RUN_GEN" "$BENCH" "$REDUCE" "$EXTRACT_NSYS" "$PROMPT"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
command -v nsys >/dev/null

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "PERF_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
git status --short --branch
sha256sum "$MODEL" "$SERVER" "$RUN_GEN" "$BENCH" "$REDUCE" "$EXTRACT_NSYS" >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

run_arm 1 default 1 8
run_arm 1 rollback 8 1
run_arm 2 rollback 1 8
run_arm 2 default 8 1
run_arm 3 default 1 8
run_arm 3 rollback 8 1
run_arm 4 rollback 1 8
run_arm 4 default 8 1
run_arm 5 default 1 8
run_arm 5 rollback 8 1

python3 "$REDUCE" "$OUT/points.jsonl" "$OUT/summary.json" | tee "$OUT/reduce.log"

# Trace observations are excluded from the N=5 medians.
run_trace default
wait_idle
run_trace rollback
wait_idle

snapshot "$OUT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "PERF_PASS $(date -u +%FT%TZ)"
