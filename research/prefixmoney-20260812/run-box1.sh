#!/usr/bin/env bash
# Deferred two-device Step-3.7 PP-2 prefix-cache battery. Do not run while box1 is occupied.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
REPO=${PREFIXMONEY_REPO:-$(git rev-parse --show-toplevel)}
LANE=$REPO/research/prefixmoney-20260812
MODEL_ROOT=${PREFIXMONEY_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${PREFIXMONEY_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${PREFIXMONEY_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
if [[ -n ${PREFIXMONEY_SERVER:-} ]]; then
    SERVER=$PREFIXMONEY_SERVER
    CUSTOM_SERVER=1
else
    SERVER=$REPO/target/release/memra-server
    CUSTOM_SERVER=0
fi
PORT=${PREFIXMONEY_PORT:-18513}
BASE=http://127.0.0.1:$PORT
STAMP=${PREFIXMONEY_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${PREFIXMONEY_OUT:-$LANE/raw/box1-$STAMP}
SERVER_PID=
SAMPLER_PID=

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
test -z "$dirty" || { echo "$dirty"; echo "FAIL: staged box1 checkout is dirty"; exit 1; }
for artifact in "$MODEL" "$DRAFT" "$LANE/prefix_gate.py" \
                "$LANE/cache_concurrency.py"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

cd "$REPO"
if [[ $CUSTOM_SERVER == 1 ]]; then
    : "${EXPECTED_SERVER_SHA256:?custom PREFIXMONEY_SERVER requires EXPECTED_SERVER_SHA256}"
    test "$(sha256sum "$SERVER" | awk '{print $1}')" = "$EXPECTED_SERVER_SHA256"
    echo "custom_server_sha256=$EXPECTED_SERVER_SHA256"
else
    set +e
    nice -n 10 ionice -c 2 -n 7 timeout 7200 \
        cargo build --release -p memra-server --bin memra-server \
        2>&1 | tee "$OUT/build-server.log"
    build_rc=${PIPESTATUS[0]}
    set -e
    echo "$build_rc" >"$OUT/build-server.exit"
    test "$build_rc" -eq 0 || { echo "FAIL: server build rc=$build_rc"; exit "$build_rc"; }
fi
test -x "$SERVER"

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
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free \
            --format=csv,noheader
        compute_apps || true
    } >"$path" 2>&1
}

stop_server() {
    if [[ -n ${SERVER_PID:-} ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 120); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

stop_sampler() {
    if [[ -n ${SAMPLER_PID:-} ]]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=
    fi
}

cleanup() {
    stop_server
    stop_sampler
}
trap cleanup EXIT INT TERM

wait_ready() {
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "FAIL: server died before readiness"
            tail -200 "$OUT/server.log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server readiness timeout"
    tail -200 "$OUT/server.log"
    return 1
}

finalize() {
    stop_server
    stop_sampler
    snapshot "$OUT/nvidia-smi-after.log" complete
    grep -Ein 'prefix-cache|dual-pp|PP-2|refused|CUDA_ERROR|out of memory|panicked' \
        "$OUT/server.log" >"$OUT/server-markers.log" || true
    # driver.log is still receiving the final verdict and lock-release lines after finalize.
    # The orchestrator seals it after the runner exits; seal every already-closed raw file here.
    find "$OUT" -maxdepth 1 -type f ! -name SHA256SUMS ! -name driver.log -print0 \
        | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS"
}

run_locked() {
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    test "$gpu_count" -ge 2 || { echo "FAIL: need two GPUs, found $gpu_count"; return 1; }
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: box1 is not GPU-idle"; return 1; }

    snapshot "$OUT/nvidia-smi-before.log" start
    sha256sum "$MODEL" "$DRAFT" "$SERVER" "$LANE/prefix_gate.py" \
        "$LANE/cache_concurrency.py" >"$OUT/SHA256SUMS.input"
    {
        echo "source_commit=$(git rev-parse HEAD)"
        echo "model=$MODEL"
        echo "draft=$DRAFT"
        echo "shape=Step-3.7 trunk+drafter loaded; plain serving; native-peer PP-2"
        echo "MEMRA_PP_STAGES=2"
        echo "MEMRA_PP_DEVICES=0,1"
        echo "MEMRA_DUAL_PP=<unset; default Auto>"
        echo "MEMRA_PP_OVERLAP=<unset; follows dual PP>"
        echo "MEMRA_PRIME_PIPE=<unset; production default>"
        echo "MEMRA_SWA_RING=0"
        echo "MEMRA_PP_HOST_BOUNCE=<unset>"
        echo "MEMRA_PREFIX_CACHE_MB=4096"
        echo "MEMRA_SERVE_SPEC=0"
        echo "exactness=N=3; K=4096; suffix=455; max_new=64; concurrent_hits=8"
        echo "capacity=N=3; c=1,2,4,8,16,24,32; K=4096; suffix=455; max_new=16"
    } >"$OUT/provenance.txt"

    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/gpu.csv" 2>&1 &
    SAMPLER_PID=$!

    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_PRIME_PIPE -u MEMRA_PP_STREAMS -u MEMRA_PP_SHARD \
        -u MEMRA_PEER_PROBE -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="step37=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SWA_RING=0 \
        MEMRA_PREFIX_CACHE_MB=4096 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
        MEMRA_AFFINITY=0 MEMRA_SERVE_SPEC=0 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_MAX_SESSIONS=64 \
        "$SERVER" >"$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready
    curl -sf "$BASE/metrics" >"$OUT/metrics-before.json"

    set +e
    timeout 14400 python3 "$LANE/prefix_gate.py" \
        --base "$BASE" --model step37 --out "$OUT/exactness.jsonl" \
        --namespace box1-prefixmoney-exact --reps 3 --prefix-tokens 4096 \
        --suffix-tokens 455 --max-tokens 64 --concurrency 8 --require-dual \
        2>&1 | tee "$OUT/exactness.log"
    local exact_rc=${PIPESTATUS[0]}
    set -e
    echo "$exact_rc" >"$OUT/exactness.exit"
    curl -sf "$BASE/metrics" >"$OUT/metrics-after-exactness.json"
    if [[ $exact_rc -ne 0 ]]; then
        echo "FAIL: exactness/refusal gate rc=$exact_rc; capacity ladder not run"
        finalize
        return "$exact_rc"
    fi

    set +e
    timeout 43200 python3 "$LANE/cache_concurrency.py" \
        --base "$BASE" --model step37 --out "$OUT/capacity.jsonl" \
        --namespace box1-prefixmoney-capacity --reps 3 \
        --concurrency 1,2,4,8,16,24,32 --prefix-tokens 4096 \
        --suffix-tokens 455 --max-tokens 16 \
        2>&1 | tee "$OUT/capacity.log"
    local capacity_rc=${PIPESTATUS[0]}
    set -e
    echo "$capacity_rc" >"$OUT/capacity.exit"
    curl -sf "$BASE/metrics" >"$OUT/metrics-after-capacity.json"
    finalize
    if [[ $capacity_rc -ne 0 ]]; then
        echo "FAIL: capacity gate rc=$capacity_rc"
        return "$capacity_rc"
    fi
    test -z "$(compute_apps)" || { compute_apps; echo "FAIL: GPU process remained"; return 1; }
    echo "PREFIXMONEY_BOX1_PASS $(date -u +%FT%TZ)"
}

if [[ ${PREFIXMONEY_LOCK_HELD:-0} == 1 ]]; then
    if ! test -e /proc/$$/fd/9 || ! flock -n 9; then
        echo "FAIL: caller lock fd 9 missing"
        exit 75
    fi
    run_locked
else
    (
        flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
        echo "lock_acquired=$(date -u +%FT%TZ)"
        set +e
        run_locked
        rc=$?
        set -e
        echo "lock_release=$(date -u +%FT%TZ)"
        exit "$rc"
    ) 9>/tmp/memra-gpu.lock
fi
