#!/usr/bin/env bash
# Single-card Q27 mixed90 runner. Build and GPU stages live in a lane-owned remote root.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
    build|run) ;;
    *) echo "usage: $0 build|run" >&2; exit 2 ;;
esac

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${KNEERAISE_ROOT:-/opt/dl-image/nvme/cx-kneeraise}
REPO=${KNEERAISE_REPO:-$ROOT/memra}
LANE=${KNEERAISE_HARNESS:-$REPO/research/kneeraise-20260812}
FROZEN=$REPO/research/sellgate-20260812
MODEL=${KNEERAISE_MODEL:-/opt/dl-image/nvme/cx-gateway/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
SERVER=$REPO/target/release/memra-server
EXPECTED_SOURCE=${KNEERAISE_EXPECTED_SOURCE:-b671c3e17035d757944439a5345b66d2f442ebe5}
EXPECTED_MODEL=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
PORT=${KNEERAISE_PORT:-18427}
BASE=http://127.0.0.1:$PORT
LABEL=${KNEERAISE_LABEL:-baseline}
STAMP=${KNEERAISE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${KNEERAISE_OUT:-$ROOT/raw/$LABEL-$STAMP}
CACHE_MB=${KNEERAISE_PREFIX_CACHE_MB:-4096}
MAX_SESSIONS=${KNEERAISE_MAX_SESSIONS:-96}
DECODE_CAP=${KNEERAISE_DECODE_BATCH_CAP:-}
PREFILL_TICK=${KNEERAISE_PREFILL_TICK:-}
TICK_TRACE=${KNEERAISE_TICK_TRACE:-0}
LEVELS=${KNEERAISE_LEVELS:-8,12,16,20,24}
REPS=${KNEERAISE_REPS:-5}
REP_START=${KNEERAISE_REP_START:-1}
SERVER_PID=
GPU_SAMPLER_PID=
DMON_PID=

source_preflight() {
    test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
    local dirty
    dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
    test -z "$dirty" || { echo "$dirty"; echo "FAIL: staged source is dirty"; return 1; }
    git -C "$REPO" merge-base --is-ancestor "$EXPECTED_SOURCE" \
        b671c3e17035d757944439a5345b66d2f442ebe5
}

compute_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,\
power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,\
utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
    } >"$path" 2>&1
}

stop_server() {
    test -n "${SERVER_PID:-}" || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server pid=$SERVER_PID did not stop after 120 seconds"
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    return 1
}

stop_samplers() {
    local pid
    for pid in "${GPU_SAMPLER_PID:-}" "${DMON_PID:-}"; do
        test -n "$pid" || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    GPU_SAMPLER_PID=
    DMON_PID=
}

cleanup() {
    stop_server || true
    stop_samplers
}

wait_ready() {
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "FAIL: server died during boot"
            tail -200 "$OUT/server.log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server readiness timeout"
    tail -200 "$OUT/server.log"
    return 1
}

build_runtime() {
    source_preflight
    test ! -e "$OUT" || { echo "FAIL: output exists: $OUT"; return 1; }
    mkdir -p "$OUT"
    exec > >(tee "$OUT/build.log") 2>&1
    echo "BUILD_START ts=$(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
    cd "$REPO"
    nice -n 10 ionice -c 2 -n 7 cargo build --release -p memra-server --bin memra-server
    sha256sum "$SERVER" >"$OUT/runtime-binaries.sha256"
    git status --porcelain --untracked-files=all >"$OUT/git-status.txt"
    test ! -s "$OUT/git-status.txt"
    touch "$OUT/build.ok"
    echo "BUILD_PASS ts=$(date -u +%FT%TZ)"
}

run_locked() {
    source_preflight
    test -x "$SERVER"
    test -f "$MODEL"
    test "$(sha256sum "$MODEL" | awk '{print $1}')" = "$EXPECTED_MODEL"
    test ! -e "$OUT" || { echo "FAIL: output exists: $OUT"; return 1; }
    mkdir -p "$OUT"
    exec > >(tee "$OUT/driver.log") 2>&1
    trap cleanup EXIT INT TERM

    snapshot "$OUT/gpu-before.log" before
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: box1 GPUs are not idle"; return 1; }
    test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2

    sha256sum "$MODEL" "$SERVER" "$LANE/sweep.py" \
        "$FROZEN/sellgate_replay.py" "$FROZEN/workload.lock.json" >"$OUT/SHA256SUMS.input"
    {
        echo "timestamp=$(date -u +%FT%TZ)"
        echo "runtime_source=$EXPECTED_SOURCE"
        echo "shape=Q27 single server on physical GPU0; GPU1 idle"
        echo "model=$MODEL"
        echo "label=$LABEL"
        echo "MEMRA_PREFIX_CACHE_MB=$CACHE_MB"
        echo "MEMRA_MAX_SESSIONS=$MAX_SESSIONS"
        echo "MEMRA_DECODE_BATCH_CAP=${DECODE_CAP:-<unset>}"
        echo "MEMRA_PREFILL_TICK=${PREFILL_TICK:-<unset>}"
        echo "MEMRA_TICK_TRACE=$TICK_TRACE"
        echo "concurrency=$LEVELS"
        echo "repetitions=$REPS"
        git -C "$REPO" log -5 --oneline --decorate
        rustc --version
        cargo --version
        nvcc --version
        nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
            --format=csv,noheader
    } >"$OUT/provenance.txt" 2>&1

    nvidia-smi --id=0 --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,\
power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
    GPU_SAMPLER_PID=$!
    nvidia-smi dmon -i 0 -s pucmt -d 1 -o DT >"$OUT/dmon-1s.log" 2>&1 &
    DMON_PID=$!

    local -a server_env
    server_env=(
        CUDA_VISIBLE_DEVICES=0
        "MEMRA_MODELS=q27=$MODEL"
        MEMRA_COMPAT=openai
        "MEMRA_ADDR=127.0.0.1:$PORT"
        "MEMRA_TAG=cx-kneeraise-$LABEL"
        MEMRA_SERVE_SPEC=0
        MEMRA_CTX=8192
        "MEMRA_PREFIX_CACHE_MB=$CACHE_MB"
        MEMRA_PREFIX_DEDUP=1
        MEMRA_REUSE_POOL=0
        MEMRA_AFFINITY=0
        "MEMRA_MAX_SESSIONS=$MAX_SESSIONS"
    )
    test -z "$DECODE_CAP" || server_env+=("MEMRA_DECODE_BATCH_CAP=$DECODE_CAP")
    test -z "$PREFILL_TICK" || server_env+=("MEMRA_PREFILL_TICK=$PREFILL_TICK")
    test "$TICK_TRACE" != 1 || server_env+=(MEMRA_TICK_TRACE=1)
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_PREFILL_TICK -u MEMRA_TICK_TRACE \
        -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
        "${server_env[@]}" "$SERVER" >"$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready
    snapshot "$OUT/gpu-server-ready.log" server-ready
    grep -q '\[worker\] q27: decode wave cap' "$OUT/server.log"
    grep -q '\[prefix-cache\] on:' "$OUT/server.log"
    curl -sf "$BASE/metrics" >"$OUT/metrics-before.json"
    curl -sf "$BASE/health" >"$OUT/health-before.json"

    set +e
    timeout 86400 python3 "$LANE/sweep.py" \
        --base "$BASE" --model q27 --out "$OUT/sweep.jsonl" \
        --workload-lock "$FROZEN/workload.lock.json" \
        --frozen-replay "$FROZEN/sellgate_replay.py" \
        --label "$LABEL" --namespace "cx-kneeraise-$LABEL" \
        --reps "$REPS" --rep-start "$REP_START" --concurrency "$LEVELS" \
        --timeout 1800 >"$OUT/sweep.log" 2>&1
    local sweep_rc=$?
    set -e
    echo "$sweep_rc" >"$OUT/sweep.exit"
    curl -sf "$BASE/metrics" >"$OUT/metrics-final.json"
    curl -sf "$BASE/health" >"$OUT/health-final.json"
    curl -sf "$BASE/yield/metrics" >"$OUT/yield-final.json"

    stop_server
    stop_samplers
    if grep -Ein 'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|MISMATCH' "$OUT/server.log"; then
        echo "FAIL: server emitted a fatal marker"
        return 1
    fi
    snapshot "$OUT/gpu-after.log" after
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU process remained"; return 1; }
    test "$sweep_rc" -eq 0
    grep -q '"verdict": "PASS"' "$OUT/sweep.jsonl"
    find "$OUT" -maxdepth 1 -type f ! -name MANIFEST.sha256 ! -name driver.log \
        -print0 | sort -z | xargs -0 sha256sum >"$OUT/MANIFEST.sha256"
    touch "$OUT/run.ok"
    echo "KNEERAISE_RUN_PASS ts=$(date -u +%FT%TZ) out=$OUT"
    trap - EXIT INT TERM
}

run_gpu() {
    echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
    fuser -v /tmp/memra-gpu.lock 2>&1 || true
    if [[ ${KNEERAISE_LOCK_HELD:-0} == 1 ]]; then
        test -e /proc/$$/fd/9 || { echo "FAIL: inherited lock fd 9 absent"; return 75; }
        run_locked
    else
        exec 9>/tmp/memra-gpu.lock
        flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; return 75; }
        echo "KNEERAISE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
        run_locked
    fi
}

case "$MODE" in
    build) build_runtime ;;
    run) run_gpu ;;
esac
