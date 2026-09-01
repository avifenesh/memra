#!/usr/bin/env bash
# Frozen single-card Q27 mixed90 runner for the cold-prefill scheduler A/B.
set -euo pipefail

test "${1:-}" = run || { echo "usage: $0 run" >&2; exit 2; }

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${COLDHOL_ROOT:-/opt/dl-image/nvme/cx-coldhol}
REPO=${COLDHOL_REPO:-$ROOT/memra}
KNEERAISE=$REPO/research/kneeraise-20260812
FROZEN=$REPO/research/sellgate-20260812
MODEL=${COLDHOL_MODEL:-/opt/dl-image/nvme/cx-requal/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
SERVER=${COLDHOL_SERVER:?set COLDHOL_SERVER to the frozen arm binary}
EXPECTED_SERVER_SHA256=${COLDHOL_EXPECTED_SERVER_SHA256:?set the frozen server sha256}
RUNTIME_SOURCE=${COLDHOL_RUNTIME_SOURCE:?set the commit used to build the server}
EXPECTED_HARNESS_SOURCE=${COLDHOL_EXPECTED_HARNESS_SOURCE:?set the checked-out harness commit}
EXPECTED_MODEL_SHA256=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
PORT=${COLDHOL_PORT:-18428}
BASE=http://127.0.0.1:$PORT
LABEL=${COLDHOL_LABEL:-candidate-smoke}
OUT=${COLDHOL_OUT:-$ROOT/raw/$LABEL-$(date -u +%Y%m%dT%H%M%SZ)}
LEVELS=${COLDHOL_LEVELS:-8,12,16,20,24}
REPS=${COLDHOL_REPS:-5}
REP_START=${COLDHOL_REP_START:-1}
PRIME_BATCH=${COLDHOL_PRIME_BATCH:-}
EXPECT_PARTIAL=${COLDHOL_EXPECT_PARTIAL:-ignore}
SERVER_PID=
GPU_SAMPLER_PID=
DMON_PID=

case "$EXPECT_PARTIAL" in yes|no|ignore) ;; *) echo "bad COLDHOL_EXPECT_PARTIAL" >&2; exit 2 ;; esac

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

source_preflight() {
    test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HARNESS_SOURCE"
    local dirty
    dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
    test -z "$dirty" || { echo "$dirty"; echo "FAIL: harness checkout is dirty"; return 1; }
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

run_locked() {
    source_preflight
    test -x "$SERVER"
    test -f "$MODEL"
    test "$(sha256sum "$SERVER" | awk '{print $1}')" = "$EXPECTED_SERVER_SHA256"
    test "$(sha256sum "$MODEL" | awk '{print $1}')" = "$EXPECTED_MODEL_SHA256"
    test ! -e "$OUT" || { echo "FAIL: output exists: $OUT"; return 1; }
    if ss -ltnH "sport = :$PORT" | grep -q .; then
        echo "FAIL: port $PORT is already listening"
        return 1
    fi
    mkdir -p "$OUT"
    exec > >(tee "$OUT/driver.log") 2>&1
    trap cleanup EXIT INT TERM

    snapshot "$OUT/gpu-before.log" before
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: box1 GPUs are not idle"; return 1; }
    test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2

    sha256sum "$MODEL" "$SERVER" "$KNEERAISE/sweep.py" \
        "$FROZEN/sellgate_replay.py" "$FROZEN/workload.lock.json" \
        >"$OUT/SHA256SUMS.input"
    {
        echo "timestamp=$(date -u +%FT%TZ)"
        echo "harness_source=$EXPECTED_HARNESS_SOURCE"
        echo "runtime_source=$RUNTIME_SOURCE"
        echo "runtime_binary=$SERVER"
        echo "runtime_binary_sha256=$EXPECTED_SERVER_SHA256"
        echo "shape=Q27 single server on physical GPU0; GPU1 idle"
        echo "model=$MODEL"
        echo "label=$LABEL"
        echo "MEMRA_PREFIX_CACHE_MB=4096"
        echo "MEMRA_PREFIX_DEDUP=1"
        echo "MEMRA_REUSE_POOL=0"
        echo "MEMRA_AFFINITY=0"
        echo "MEMRA_MAX_SESSIONS=96"
        echo "MEMRA_SERVE_SPEC=0"
        echo "MEMRA_DECODE_BATCH_CAP=<unset>"
        echo "MEMRA_PREFILL_TICK=<unset>"
        echo "MEMRA_PRIME_BATCH=${PRIME_BATCH:-<unset>}"
        echo "MEMRA_TICK_TRACE=<unset>"
        echo "concurrency=$LEVELS"
        echo "repetitions=$REPS"
        echo "rep_start=$REP_START"
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
        "MEMRA_TAG=cx-coldhol-$LABEL"
        MEMRA_SERVE_SPEC=0
        MEMRA_CTX=8192
        MEMRA_PREFIX_CACHE_MB=4096
        MEMRA_PREFIX_DEDUP=1
        MEMRA_REUSE_POOL=0
        MEMRA_AFFINITY=0
        MEMRA_MAX_SESSIONS=96
    )
    test -z "$PRIME_BATCH" || server_env+=("MEMRA_PRIME_BATCH=$PRIME_BATCH")
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_PREFILL_TICK -u MEMRA_TICK_TRACE \
        -u MEMRA_PRIME_BATCH -u MEMRA_PRIME_BATCH_MAX_T -u MEMRA_PRIME_BATCH_HOLD_MS \
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
    timeout 86400 python3 "$KNEERAISE/sweep.py" \
        --base "$BASE" --model q27 --out "$OUT/sweep.jsonl" \
        --workload-lock "$FROZEN/workload.lock.json" \
        --frozen-replay "$FROZEN/sellgate_replay.py" \
        --label "$LABEL" --namespace "cx-coldhol-$LABEL" \
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
    awk '
        /^\[prime-batch\] B=/ { batches++ }
        /partial=[1-9][0-9]*/ { partial++ }
        /^\[prime-batch\] failed/ { failed++ }
        END {
            printf "batch_lines=%d\npartial_batch_lines=%d\nfailed_batch_lines=%d\n", batches, partial, failed
        }
    ' "$OUT/server.log" >"$OUT/prime-batch-summary.txt"
    if grep -Ein 'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|MISMATCH|\[prime-batch\] failed' "$OUT/server.log"; then
        echo "FAIL: server emitted a fatal marker"
        return 1
    fi
    case "$EXPECT_PARTIAL" in
        yes) grep -Eq '^partial_batch_lines=[1-9][0-9]*$' "$OUT/prime-batch-summary.txt" ;;
        no) grep -q '^partial_batch_lines=0$' "$OUT/prime-batch-summary.txt" ;;
    esac
    snapshot "$OUT/gpu-after.log" after
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU process remained"; return 1; }
    test "$sweep_rc" -eq 0
    grep -q '"verdict": "PASS"' "$OUT/sweep.jsonl"
    find "$OUT" -maxdepth 1 -type f ! -name MANIFEST.sha256 ! -name driver.log \
        -print0 | sort -z | xargs -0 sha256sum >"$OUT/MANIFEST.sha256"
    touch "$OUT/run.ok"
    echo "COLDHOL_RUN_PASS ts=$(date -u +%FT%TZ) out=$OUT"
    trap - EXIT INT TERM
}

echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
fuser -v /tmp/memra-gpu.lock 2>&1 || true
if [[ ${COLDHOL_LOCK_HELD:-0} == 1 ]]; then
    test -e /proc/$$/fd/9 || { echo "FAIL: inherited lock fd 9 absent"; exit 75; }
    run_locked
else
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
    echo "COLDHOL_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
    run_locked
fi
