#!/usr/bin/env bash
# Run one bounded new-box receipt block, restoring production serving and soak on every exit.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
    build|matrix|correctness|perf|capacity) ;;
    *) echo "usage: $0 build|matrix|correctness|perf|capacity" >&2; exit 2 ;;
esac

export PATH="/root/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-/workspace/memra}
HARNESS=${HARNESS:-/workspace/newboxgates-20260811/harness}
RAW_ROOT=${RAW_ROOT:-/workspace/newboxgates-20260811/raw}
STAMP=${NEWBOX_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${NEWBOX_OUT:-$RAW_ROOT/$MODE-$STAMP}
SERVER=$REPO/target/release/memra-server
KERNEL=$REPO/target/release/kernel-check
DECODE_BATCH=$REPO/target/release/decode-batch-gate
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
MODEL_ROOT=/workspace/models/step-3.7-flash
MODEL=$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf
MODEL_NAME=stepfun/step-3.7-flash
GOLDEN=$REPO/research/darktrain2-20260810/raw/qos/golden-response.bin
PROMPT_SMALL=$REPO/tools/fast-gate/prompts/probe.txt
PROMPT_4K=$REPO/research/step-sku-20260807/prompt-pp4096.txt
PROMPT_6257=$REPO/research/chunk-invariance-20260805/prompt-pp6257.txt
EXPECTED_SOURCE=${EXPECTED_SOURCE:-5911de40f48d1d2fe36a92fef7b9b41cebc792f2}
EXPECTED_SERVER=${EXPECTED_SERVER:-1b2159b50c9bb5cf2703e9f159ac44b6f40d339db6fa078a8a0212ce1d54bf7b}
EXPECTED_GOLDEN=${EXPECTED_GOLDEN:-21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de}
MEMRA_PROCESS_PATTERN='(^|/)memra-server$'
TEST_SERVER_PID=0
SAMPLER_PID=0
SERVICE_STOPPED=0

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
            --query-gpu=index,name,uuid,driver_version,pci.bus_id,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

wait_idle() {
    for _ in $(seq 1 240); do
        [[ -z $(compute_apps) ]] && return 0
        sleep 1
    done
    compute_apps
    return 1
}

wait_pid_exit() {
    local pid=$1
    for _ in $(seq 1 180); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    return 1
}

stop_named_processes() {
    local pattern=$1 pids pid
    pids=$(pgrep -f "$pattern" || true)
    [[ -z $pids ]] && return 0
    for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in $pids; do
        if ! wait_pid_exit "$pid"; then
            echo "force_stopping pid=$pid pattern=$pattern"
            kill -KILL "$pid" 2>/dev/null || true
            wait_pid_exit "$pid" || return 1
        fi
    done
}

pinned_preflight() {
    local source server_hash golden_hash
    source=$(git -C "$REPO" rev-parse HEAD)
    server_hash=$(sha256sum "$SERVER" | awk '{print $1}')
    golden_hash=$(sha256sum "$GOLDEN" | awk '{print $1}')
    echo "source_commit=$source"
    echo "server_sha256=$server_hash"
    echo "golden_sha256=$golden_hash"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT" "$GOLDEN"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $server_hash == "$EXPECTED_SERVER" ]]
    [[ $golden_hash == "$EXPECTED_GOLDEN" ]]
}

capture_service_state() {
    local root=$1
    mkdir -p "$root"
    date -u +%FT%TZ >"$root/timestamp.txt"
    pgrep -af 'memra-server|/root/soak.py' >"$root/processes.txt" || true
    curl -fsS http://127.0.0.1:8002/v1/models >"$root/models.json" 2>"$root/models.err" || true
    tail -200 /var/log/memra-server.log >"$root/server-tail.log" 2>&1 || true
    tail -50 /var/log/soak.jsonl >"$root/soak-tail.jsonl" 2>&1 || true
    snapshot "$root/gpus.log" service-state
}

stop_service() {
    SERVICE_STOPPED=1
    capture_service_state "$OUT/service-before"
    stop_named_processes '^python3 /root/soak\.py$'
    stop_named_processes "$MEMRA_PROCESS_PATTERN"
    wait_idle
    snapshot "$OUT/service-stopped-gpus.log" service-stopped
    echo "production_service_stopped=$(date -u +%FT%TZ)"
}

verify_stream() {
    local root=$1
    curl --no-buffer -fsS --max-time 300 \
        -H 'Content-Type: application/json' \
        -d '{"model":"stepfun/step-3.7-flash","messages":[{"role":"user","content":"Reply with exactly: NEWBOX OK"}],"max_tokens":32,"temperature":0,"seed":3407,"stream":true,"stream_options":{"include_usage":true}}' \
        http://127.0.0.1:8002/v1/chat/completions >"$root/streamed-completion.sse"
    python3 - "$root/streamed-completion.sse" "$root/streamed-completion-summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
pieces = []
usage = {}
done = False
for line in source.read_text(errors="replace").splitlines():
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        done = True
        continue
    event = json.loads(payload)
    if event.get("error"):
        raise SystemExit(event["error"])
    usage = event.get("usage") or usage
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        pieces.append(
            (delta.get("content") or "")
            + (delta.get("reasoning") or "")
            + (delta.get("reasoning_content") or "")
        )
text = "".join(pieces)
receipt = {
    "done": done,
    "visible_text": text,
    "visible_bytes": len(text.encode()),
    "visible_sha256": hashlib.sha256(text.encode()).hexdigest(),
    "usage": usage,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
assert done and text.strip(), receipt
PY
}

restore_service() {
    local root=$OUT/service-restored
    mkdir -p "$root"
    stop_named_processes '^python3 /root/soak\.py$' || true
    stop_named_processes "$MEMRA_PROCESS_PATTERN" || true
    wait_idle || return 1
    cd "$REPO"
    setsid nohup /root/start-memra.sh >/var/log/memra-server.log 2>&1 </dev/null &
    echo $! >"$root/server-launch.pid"
    for _ in $(seq 1 900); do
        if curl -fsS http://127.0.0.1:8002/readyz >/dev/null 2>&1; then break; fi
        if ! kill -0 "$(cat "$root/server-launch.pid")" 2>/dev/null; then
            tail -200 /var/log/memra-server.log >"$root/server-start-failure.log" 2>&1 || true
            return 1
        fi
        sleep 1
    done
    curl -fsS http://127.0.0.1:8002/readyz >"$root/readyz.txt"
    curl -fsS http://127.0.0.1:8002/v1/models >"$root/models.json"
    verify_stream "$root"
    setsid nohup python3 /root/soak.py >/var/log/soak-driver.log 2>&1 </dev/null &
    echo $! >"$root/soak-launch.pid"
    sleep 2
    kill -0 "$(cat "$root/soak-launch.pid")"
    pgrep -af 'memra-server|/root/soak.py' >"$root/processes.txt"
    snapshot "$root/gpus.log" service-restored
    tail -200 /var/log/memra-server.log >"$root/server-tail.log" 2>&1 || true
    touch "$root/restored.ok"
    echo "production_service_restored=$(date -u +%FT%TZ)"
}

stop_test_server() {
    if (( TEST_SERVER_PID > 0 )); then
        kill -TERM "$TEST_SERVER_PID" 2>/dev/null || true
        wait "$TEST_SERVER_PID" 2>/dev/null || true
        TEST_SERVER_PID=0
    fi
}

stop_sampler() {
    if (( SAMPLER_PID > 0 )); then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=0
    fi
}

on_exit() {
    local rc=$? restore_rc=0
    trap - EXIT INT TERM
    set +e
    stop_sampler
    stop_test_server
    if (( SERVICE_STOPPED )); then restore_service; restore_rc=$?; fi
    if (( rc == 0 && restore_rc != 0 )); then rc=$restore_rc; fi
    echo "block=$MODE out=$OUT exit=$rc restore_exit=$restore_rc done=$(date -u +%FT%TZ)"
    exit "$rc"
}
trap on_exit EXIT INT TERM

run_logged() {
    local label=$1 log=$2 rc
    shift 2
    echo "gate=$label start=$(date -u +%FT%TZ)"
    set +e
    timeout 14400 "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/correctness/$label.rc"
    [[ $rc -eq 0 ]]
    wait_idle
    echo "gate=$label done=$(date -u +%FT%TZ)"
}

run_build() {
    pinned_preflight
    capture_service_state "$OUT/service-during-build"
    {
        rustc --version
        cargo --version
        nvcc --version
    } >"$OUT/toolchain.txt" 2>&1
    cd "$REPO"
    cargo build --release -p memra-engine \
        --bin kernel-check --bin decode-batch-gate --bin run-gen --bin run-spec \
        --bin concat-prime-probe 2>&1 | tee "$OUT/cargo-build.log"
    sha256sum "$SERVER" "$KERNEL" "$DECODE_BATCH" "$RUN_GEN" "$RUN_SPEC" \
        "$REPO/target/release/concat-prime-probe" >"$OUT/binary-sha256.txt"
    [[ $(sha256sum "$SERVER" | awk '{print $1}') == "$EXPECTED_SERVER" ]]
    capture_service_state "$OUT/service-after-build"
    touch "$OUT/build.ok"
}

run_matrix() {
    stop_service
    pinned_preflight
    local common=(
        REPO="$REPO" BIN="$SERVER" WORK_ROOT="$OUT"
        QOS="$REPO/research/p0iso-20260810/qos_probe.py"
        MODEL_ROOT="$MODEL_ROOT" MODEL="$MODEL" DRAFT="$DRAFT"
        MODEL_NAME="$MODEL_NAME" GOLDEN="$GOLDEN" PORT=18431
        RUN_ROOT="$OUT/matrix" EXPECTED_SOURCE="$EXPECTED_SOURCE"
        EXPECTED_BINARY="$EXPECTED_SERVER" EXPECTED_GOLDEN="$EXPECTED_GOLDEN"
    )
    env "${common[@]}" "$REPO/research/p0iso-20260810/run-box1.sh" h2-c1 10
    env "${common[@]}" "$REPO/research/p0iso-20260810/run-box1.sh" same 5
    python3 "$HARNESS/reduce_receipts.py" matrix "$OUT/matrix" "$OUT/matrix-summary.json"
}

run_correctness() {
    stop_service
    pinned_preflight
    mkdir -p "$OUT/correctness"
    exec 9>/tmp/memra-gpu.lock
    flock -w 60 9 || { echo LOCK_TIMEOUT; return 75; }
    wait_idle
    snapshot "$OUT/correctness/nvidia-smi-before.log" correctness-preflight

    run_logged kernel-check "$OUT/correctness/kernel-check.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 "$KERNEL" "$MODEL"
    grep -q '^ALL GREEN ([0-9][0-9]* cells, [0-9][0-9]* skipped)$' \
        "$OUT/correctness/kernel-check.log"

    run_logged decode-batch-gate "$OUT/correctness/decode-batch-gate.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
        "$DECODE_BATCH" "$MODEL" --mode pp --batch 1,2,4,8 --steps 24 --reps 2 \
        --stages 2 --plen 520
    grep -q 'pp mode verdict: 0 failing arm(s)' "$OUT/correctness/decode-batch-gate.log"

    run_logged run-gen "$OUT/correctness/run-gen.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_NGEN=64 MEMRA_PROMPT_FILE="$PROMPT_SMALL" "$RUN_GEN" "$MODEL"
    grep -q 'prefill argmax=.*decode argmax=.*MATCH' "$OUT/correctness/run-gen.log"
    grep -q 'batched-prime argmax=.*tokenwise argmax=.*MATCH' "$OUT/correctness/run-gen.log"

    run_logged run-spec "$OUT/correctness/run-spec.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT_SMALL" \
        "$RUN_SPEC" "$MODEL"
    [[ $(grep -c 'self-consistency: PASS' "$OUT/correctness/run-spec.log") -eq 8 ]]
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/correctness/run-spec.log"

    run_logged chunk-naked "$OUT/correctness/chunk-naked.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_CHUNKINV_LOG="$OUT/correctness/chunk-naked-raw.log" \
        "$REPO/tools/chunk-invariance-gate.sh" "$MODEL" --label step35-swa \
        --prompts "$PROMPT_6257" --chunks 4096,513,512,256,64 \
        --seam MEMRA_STEP35_SWA_TKV --steps 24

    run_logged chunk-canary "$OUT/correctness/chunk-canary.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_CHUNKINV_LOG="$OUT/correctness/chunk-canary-raw.log" \
        "$REPO/tools/chunk-invariance-gate.sh" "$MODEL" --label step35-swa \
        --prompts "$PROMPT_6257" --chunks 4096,513,512,256,64 \
        --seam MEMRA_STEP35_SWA_TKV --steps 24 --canary

    run_logged tick-naked "$OUT/correctness/tick-naked.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        "$REPO/tools/tick-invariance-gate.sh" "$MODEL" --label step35-tick \
        --prompts "$PROMPT_6257" --budgets 0,1024,513,512,256,64 \
        --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24
    local tick_raw
    tick_raw=$(grep -oE '/tmp/tickinv-gate-[^ )]+\.log' "$OUT/correctness/tick-naked.log" | tail -1)
    cp "$tick_raw" "$OUT/correctness/tick-naked-raw.log"

    run_logged tick-canary "$OUT/correctness/tick-canary.log" \
        env -u MEMRA_SWA_RING CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        "$REPO/tools/tick-invariance-gate.sh" "$MODEL" --label step35-tick \
        --prompts "$PROMPT_6257" --budgets 0,1024,513,512,256,64 \
        --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24 --canary
    tick_raw=$(grep -oE '/tmp/tickinv-gate-[^ )]+\.log' "$OUT/correctness/tick-canary.log" | tail -1)
    cp "$tick_raw" "$OUT/correctness/tick-canary-raw.log"

    snapshot "$OUT/correctness/nvidia-smi-after.log" correctness-final
    python3 "$HARNESS/reduce_receipts.py" correctness "$OUT/correctness" \
        "$OUT/correctness-summary.json"
}

wait_test_ready() {
    local base=$1 log=$2
    for _ in $(seq 1 900); do
        curl -fsS "$base/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$TEST_SERVER_PID" 2>/dev/null; then tail -200 "$log"; return 1; fi
        sleep 1
    done
    tail -200 "$log"
    return 1
}

start_perf_server() {
    local rep=$1 log=$2 port=$3
    env -u MEMRA_SWA_RING -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_BG_JOB \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$port" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_PREFIX_CACHE_MB=2048 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_TAG="newbox-perf-r$rep" "$SERVER" >"$log" 2>&1 &
    TEST_SERVER_PID=$!
    wait_test_ready "http://127.0.0.1:$port" "$log"
}

run_perf_point() {
    local point=$1 rep=$2 base=$3
    local label="r${rep}-${point}" log="$OUT/client-r${rep}-${point}.log"
    local -a args=(--base "$base" --model "$MODEL_NAME" --label "$label" \
        --out "$OUT/points.jsonl" --timeout 1800)
    case "$point" in
        short) args+=(--shape short --concurrency 1 --max-tokens 32) ;;
        4k) args+=(--shape 4k --concurrency 1 --max-tokens 8 --prompt-file "$PROMPT_4K" \
            --expect-prompt-tokens 4107) ;;
        c1) args+=(--shape decode --concurrency 1 --max-tokens 512 --require-length) ;;
        c4) args+=(--shape decode --concurrency 4 --max-tokens 512 --require-length) ;;
        c8) args+=(--shape decode --concurrency 8 --max-tokens 512 --require-length) ;;
    esac
    python3 "$HARNESS/serve_bench.py" "${args[@]}" 2>&1 | tee "$log"
}

run_perf() {
    stop_service
    pinned_preflight
    exec 9>/tmp/memra-gpu.lock
    flock -w 60 9 || { echo LOCK_TIMEOUT; return 75; }
    wait_idle
    snapshot "$OUT/nvidia-smi-before.log" perf-preflight
    : >"$OUT/points.jsonl"
    local rep order point port=18432 base=http://127.0.0.1:18432 server_log
    for rep in $(seq 1 5); do
        snapshot "$OUT/thermal-r${rep}-before.log" "perf-r${rep}-before"
        server_log="$OUT/server-r${rep}.log"
        start_perf_server "$rep" "$server_log" "$port"
        nvidia-smi \
            --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits -lms 500 >"$OUT/gpu-r${rep}.csv" 2>&1 &
        SAMPLER_PID=$!
        python3 "$HARNESS/serve_bench.py" --base "$base" --model "$MODEL_NAME" \
            --shape warmup --label "r${rep}-warmup" --out "$OUT/warmups.jsonl" \
            --concurrency 1 --max-tokens 16 --timeout 1800 \
            >"$OUT/client-r${rep}-warmup.log" 2>&1
        if (( rep % 2 == 1 )); then order="short 4k c1 c4 c8"; else order="c8 c4 c1 4k short"; fi
        for point in $order; do run_perf_point "$point" "$rep" "$base"; done
        stop_sampler
        stop_test_server
        wait_idle
        if grep -Ein 'CUDA_ERROR|out of memory|panicked at|worker.*died|server fatal' "$server_log"; then
            echo "FATAL: server failure signature in $server_log"
            return 1
        fi
        snapshot "$OUT/thermal-r${rep}-after.log" "perf-r${rep}-after"
    done
    python3 "$HARNESS/reduce_receipts.py" bench "$OUT/points.jsonl" "$OUT/perf-summary.json"
    snapshot "$OUT/nvidia-smi-after.log" perf-final
}

run_capacity() {
    local target_path="$REPO/target"
    local workload_path="$REPO/research/capbase-20260809/run_workloads.py"
    local analyzer_path="$REPO/research/capbase-20260809/analyze_capacity.py"
    local capacity_script="$REPO/research/ringval-20260810/run-box1-capacity.sh"
    stop_service
    pinned_preflight
    RINGVAL_STAMP="$STAMP" RINGVAL_OUT="$OUT/capacity" \
        REPO="$REPO" TARGET="$target_path" BIN="$SERVER" \
        WORKLOAD="$workload_path" ANALYZER="$analyzer_path" \
        MODEL_ROOT="$MODEL_ROOT" MODEL="$MODEL" DRAFT="$DRAFT" PORT=18433 \
        EXPECTED_SOURCE="$EXPECTED_SOURCE" EXPECTED_BINARY="$EXPECTED_SERVER" \
        "$capacity_script"
}

echo "block=$MODE out=$OUT start=$(date -u +%FT%TZ) host=$(hostname)"
case "$MODE" in
    build) run_build ;;
    matrix) run_matrix ;;
    correctness) run_correctness ;;
    perf) run_perf ;;
    capacity) run_capacity ;;
esac
