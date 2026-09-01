#!/usr/bin/env bash
# One-lock local RTX 5090 eager-arm campaign. Never changes GPU clocks.
set -Eeu -o pipefail

: "${EXPECTED_BINARY_SHA:?set EXPECTED_BINARY_SHA to the sealed memra-server SHA-256}"
REPO=$(git rev-parse --show-toplevel)
LANE=$REPO/research/armrig5090-20260813
OUT=${ARMRIG_OUT:-$LANE/raw/attempt2-local5090}
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
SERVER=$REPO/target/release/memra-server
FULL_BENCH=$LANE/full_hit_cell.py
MIXED_BENCH=$LANE/mixed_cell.py
PAIR_GATE=$LANE/pair_gate.py
REDUCE=$LANE/reduce.py
SELLGATE_MODULE=$REPO/research/sellgate-20260812/sellgate_replay.py
WORKLOAD_LOCK=$REPO/research/sellgate-20260812/workload.lock.json
BINARY_SOURCE=${BINARY_SOURCE:-57ebcf8d319dc8ea9bb351b39fc1ab28d18c20db}
PORT=${ARMRIG_PORT:-18469}
BASE=http://127.0.0.1:$PORT
LOCK=/tmp/memra-5090.lock

for artifact in "$MODEL" "$SERVER" "$FULL_BENCH" "$MIXED_BENCH" "$PAIR_GATE" \
    "$REDUCE" "$SELLGATE_MODULE" "$WORKLOAD_LOCK"; do
    test -f "$artifact" || { echo "FAIL: missing artifact $artifact"; exit 1; }
done
git merge-base --is-ancestor "$BINARY_SOURCE" HEAD || {
    echo "FAIL: binary source $BINARY_SOURCE is not an ancestor of HEAD"
    exit 1
}
git diff --quiet "$BINARY_SOURCE" -- Cargo.toml Cargo.lock crates || {
    echo "FAIL: runtime source changed after the sealed build"
    exit 1
}
test "$(sha256sum "$SERVER" | awk '{print $1}')" = "$EXPECTED_BINARY_SHA" || {
    sha256sum "$SERVER"
    echo "FAIL: server binary does not match EXPECTED_BINARY_SHA=$EXPECTED_BINARY_SHA"
    exit 1
}
dirty=$(git status --porcelain --untracked-files=all)
test -z "$dirty" || { echo "$dirty"; echo "FAIL: worktree must be clean before scoring"; exit 1; }
test ! -e "$OUT" || { echo "FAIL: refusing to overwrite $OUT"; exit 1; }

mkdir -p "$OUT/full" "$OUT/mixed" "$OUT/activation" "$OUT/qualification" \
    "$OUT/exactness"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=
sampler_pid=
lock_acquired=0
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | tr -d '[:space:]')

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
        ss -ltnp | grep -E ":${PORT}\\b" || true
    } >"$path" 2>&1
}

wait_idle() {
    local apps
    for _ in $(seq 1 180); do
        apps=$(compute_apps)
        test -z "$apps" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

assert_idle() {
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU compute applications present"; return 1; }
}

assert_port_clear() {
    if ss -ltn 2>/dev/null | grep -qE ":${PORT}[[:space:]]"; then
        ss -ltnp 2>/dev/null | grep -E ":${PORT}[[:space:]]" || true
        echo "FAIL: port $PORT is occupied"
        return 1
    fi
}

assert_owned_server() {
    local apps pid uuid bad=0 count=0
    apps=$(compute_apps)
    test -n "$apps" || { echo "FAIL: server has no visible CUDA context"; return 1; }
    while IFS=, read -r pid uuid _; do
        pid=$(echo "$pid" | xargs)
        uuid=$(echo "$uuid" | xargs)
        count=$((count + 1))
        if [[ $pid != "$server_pid" || $uuid != "$GPU_UUID" ]]; then
            bad=1
        fi
    done <<<"$apps"
    if (( bad != 0 || count != 1 )); then
        echo "$apps"
        echo "FAIL: compute census is not exactly owned server pid=$server_pid"
        return 1
    fi
}

stop_sampler() {
    local pid=${sampler_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sampler_pid=
}

stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            wait_idle
            return 0
        fi
        sleep 1
    done
    echo "FAIL: owned server pid=$pid did not stop after 120 seconds"
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    server_pid=
    wait_idle || true
    return 1
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    stop_server
    stop_sampler
    snapshot "$OUT/cleanup-trap.log" "trap-rc-$rc"
    if (( lock_acquired == 1 )); then
        flock -u 9
        exec 9>&-
    fi
    echo "CAMPAIGN_EXIT rc=$rc ts=$(date -u +%FT%TZ)"
    exit "$rc"
}
trap cleanup EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "FAIL: server died during boot"
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server did not become ready"
    tail -200 "$log"
    return 1
}

assert_server_clean() {
    local log=$1 failures
    failures=$(grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|sentinel' \
        "$log" || true)
    if [[ -n $failures ]]; then
        echo "$failures"
        echo "FAIL: server failure signature in $log"
        return 1
    fi
}

run_logged() {
    local log=$1
    shift
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    return "$rc"
}

start_server() {
    local arm=$1 label=$2 census=${3:-0}
    local log="$OUT/$label-server.log"
    local env_path="$OUT/$label-env.txt"
    local current_sha
    current_sha=$(sha256sum "$SERVER" | awk '{print $1}')
    test "$current_sha" = "$EXPECTED_BINARY_SHA" || {
        echo "FAIL: binary changed before $label: $current_sha"
        return 1
    }
    local -a policy=()
    if [[ $arm == eager ]]; then
        policy=(MEMRA_SERVE_B1FAST=1 MEMRA_SERVE_GS=1)
    elif [[ $arm != repaired ]]; then
        echo "FAIL: unknown policy arm $arm"
        return 1
    fi
    if (( census == 1 )); then
        policy+=(MEMRA_GRAPH_CENSUS=1)
    fi
    assert_idle
    assert_port_clear
    {
        echo "label=$label"
        echo "model=q27"
        echo "model_path=$MODEL"
        echo "policy_arm=$arm"
        echo "binary=$SERVER"
        echo "binary_sha256=$current_sha"
        if [[ $arm == eager ]]; then
            echo "MEMRA_SERVE_B1FAST=1"
            echo "MEMRA_SERVE_GS=1"
        else
            echo "MEMRA_SERVE_B1FAST=<unset>"
            echo "MEMRA_SERVE_GS=<unset>"
        fi
        echo "MEMRA_GS_MIN=<unset; default 384>"
        echo "MEMRA_SERVE_SPEC=0"
        echo "MEMRA_PREFIX_CACHE_MB=4096"
        echo "MEMRA_PREFIX_DEDUP=1"
        echo "MEMRA_REUSE_POOL=0"
        echo "MEMRA_AFFINITY=0"
        echo "MEMRA_CTX=8192"
        echo "MEMRA_MAX_SESSIONS=32"
        echo "MEMRA_GRAPH_CENSUS=$([[ $census == 1 ]] && echo 1 || echo '<unset>')"
    } >"$env_path"

    env -i PATH="$PATH" HOME="$HOME" TMPDIR=/home/avifenesh/tmp-lanes \
        RUST_BACKTRACE=1 CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="q27=$MODEL" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_CTX=8192 MEMRA_PREFIX_CACHE_MB=4096 MEMRA_PREFIX_DEDUP=1 \
        MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=32 \
        "${policy[@]}" "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
    assert_owned_server
    grep -q '\[prefix-cache\] on:' "$log"
}

run_activation_probe() {
    local label=activation-q27-eager
    echo "ACTIVATION_START ts=$(date -u +%FT%TZ)"
    start_server eager "activation/$label" 1
    run_logged "$OUT/activation/$label-load.log" python3 "$FULL_BENCH" \
        --base "$BASE" --model q27 --target q27 --policy-arm eager --rep 0 \
        --concurrency 1 --max-tokens 512 --label "$label" \
        --namespace "armrig5090-$label" --out "$OUT/activation-points.jsonl" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/activation/$label-metrics.json"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/activation/$label-server.log"
    grep -q '\[graph-census\]' "$OUT/activation/$label-server.log" || {
        echo "FAIL: EAGER activation probe did not capture a GraphSession"
        return 1
    }
    echo "ACTIVATION_PASS ts=$(date -u +%FT%TZ)"
}

run_width_qualification() {
    local label=qualification-q27-c16-eager
    echo "QUALIFICATION_START concurrency=16 ts=$(date -u +%FT%TZ)"
    start_server eager "qualification/$label"
    run_logged "$OUT/qualification/$label-load.log" python3 "$FULL_BENCH" \
        --base "$BASE" --model q27 --target q27 --policy-arm eager --rep 0 \
        --concurrency 16 --max-tokens 512 --label "$label" \
        --namespace "armrig5090-$label" --out "$OUT/qualification-points.jsonl" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/qualification/$label-metrics.json"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/qualification/$label-server.log"
    echo "QUALIFICATION_PASS concurrency=16 ts=$(date -u +%FT%TZ)"
}

run_full_point() {
    local concurrency=$1 rep=$2 arm=$3
    local label="full-q27-c$concurrency-r$rep-$arm"
    echo "FULL_POINT_START label=$label ts=$(date -u +%FT%TZ)"
    start_server "$arm" "full/$label"
    curl -sf "$BASE/metrics" >"$OUT/full/$label-metrics-before.json"
    snapshot "$OUT/full/$label-gpu-before.log" "$label-before"
    run_logged "$OUT/full/$label-load.log" python3 "$FULL_BENCH" \
        --base "$BASE" --model q27 --target q27 --policy-arm "$arm" --rep "$rep" \
        --concurrency "$concurrency" --max-tokens 512 --label "$label" \
        --namespace "armrig5090-$label" --out "$OUT/full-points.jsonl" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/full/$label-metrics-after.json"
    snapshot "$OUT/full/$label-gpu-after.log" "$label-after"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/full/$label-server.log"
    echo "FULL_POINT_PASS label=$label ts=$(date -u +%FT%TZ)"
}

run_mixed_point() {
    local rep=$1 arm=$2
    local label="mixed-q27-c4-r$rep-$arm"
    echo "MIXED_POINT_START label=$label ts=$(date -u +%FT%TZ)"
    start_server "$arm" "mixed/$label"
    snapshot "$OUT/mixed/$label-gpu-before.log" "$label-before"
    run_logged "$OUT/mixed/$label-load.log" python3 "$MIXED_BENCH" \
        --base "$BASE" --model q27 --target q27 --policy-arm "$arm" --rep "$rep" \
        --concurrency 4 --label "$label" --namespace "armrig5090-$label" \
        --out "$OUT/mixed-points.jsonl" --module "$SELLGATE_MODULE" \
        --workload-lock "$WORKLOAD_LOCK" --timeout 1800
    curl -sf "$BASE/metrics" >"$OUT/mixed/$label-metrics-final.json"
    snapshot "$OUT/mixed/$label-gpu-after.log" "$label-after"
    assert_owned_server
    stop_server
    assert_server_clean "$OUT/mixed/$label-server.log"
    echo "MIXED_POINT_PASS label=$label ts=$(date -u +%FT%TZ)"
}

gate_pair() {
    local kind=$1 concurrency=$2 rep=$3 path=$4
    local left right
    if [[ $kind == full ]]; then
        left="full-q27-c$concurrency-r$rep-repaired"
        right="full-q27-c$concurrency-r$rep-eager"
    else
        left="mixed-q27-c4-r$rep-repaired"
        right="mixed-q27-c4-r$rep-eager"
    fi
    local rc=0
    run_logged "$OUT/exactness/$kind-c$concurrency-r$rep.log" python3 "$PAIR_GATE" \
        --path "$path" --kind "$kind" --left-label "$left" --right-label "$right" \
        --concurrency "$concurrency" || rc=$?
    if (( rc == 2 )); then
        echo "PAIR_INVALID kind=$kind concurrency=$concurrency rep=$rep reason=BYTE_MISMATCH"
        return 0
    fi
    return "$rc"
}

echo "CAMPAIGN_START ts=$(date -u +%FT%TZ) host=$(hostname)"
echo "repo=$REPO out=$OUT binary_source=$BINARY_SOURCE"
python3 - "$FULL_BENCH" "$MIXED_BENCH" "$PAIR_GATE" "$REDUCE" "$SELLGATE_MODULE" <<'PY'
import ast
import pathlib
import sys
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
sha256sum "$MODEL" "$SERVER" "$FULL_BENCH" "$MIXED_BENCH" "$PAIR_GATE" \
    "$REDUCE" "$SELLGATE_MODULE" "$WORKLOAD_LOCK" "$0" >"$OUT/SHA256SUMS.input"
{
    echo "REPAIRED: MEMRA_SERVE_B1FAST=<unset> MEMRA_SERVE_GS=<unset>"
    echo "EAGER: MEMRA_SERVE_B1FAST=1 MEMRA_SERVE_GS=1"
    echo "Both arms execute $SERVER"
    echo "Binary SHA-256: $EXPECTED_BINARY_SHA"
} >"$OUT/arm-invariant.txt"

echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
fuser -v "$LOCK" 2>&1 || true
exec 9>"$LOCK"
flock -w 60 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
lock_acquired=1
lock_start=$(date -u +%FT%TZ)
echo "GPU_LOCK_ACQUIRED ts=$lock_start pid=$$"
assert_idle
assert_port_clear
snapshot "$OUT/gpu-before.log" lock-acquired
nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
sampler_pid=$!

run_activation_probe
run_width_qualification

widths=(1 4 16)
for rep in $(seq 1 5); do
    offset=$(( (rep - 1) % ${#widths[@]} ))
    for point in $(seq 0 $((${#widths[@]} - 1))); do
        index=$(( (point + offset) % ${#widths[@]} ))
        concurrency=${widths[$index]}
        if (( rep % 2 == 1 )); then
            run_full_point "$concurrency" "$rep" repaired
            run_full_point "$concurrency" "$rep" eager
        else
            run_full_point "$concurrency" "$rep" eager
            run_full_point "$concurrency" "$rep" repaired
        fi
        gate_pair full "$concurrency" "$rep" "$OUT/full-points.jsonl"
    done
done

# Keep the full-hit ladder complete even if the corruption class invalidates the
# later mixed90 shape. Both sections remain inside the same lock/thermal window.
for rep in $(seq 1 5); do
    if (( rep % 2 == 1 )); then
        run_mixed_point "$rep" repaired
        run_mixed_point "$rep" eager
    else
        run_mixed_point "$rep" eager
        run_mixed_point "$rep" repaired
    fi
    gate_pair mixed 4 "$rep" "$OUT/mixed-points.jsonl"
done

stop_sampler
run_logged "$OUT/reduce.log" python3 "$REDUCE" \
    --full "$OUT/full-points.jsonl" --mixed "$OUT/mixed-points.jsonl" \
    --thermal "$OUT/gpu-250ms.csv" --source "$BINARY_SOURCE" \
    --binary-sha256 "$EXPECTED_BINARY_SHA" --out "$OUT/summary.json"

assert_idle
assert_port_clear
snapshot "$OUT/gpu-after.log" campaign-complete
lock_end=$(date -u +%FT%TZ)
{
    echo "lock_acquired=$lock_start"
    echo "lock_released=$lock_end"
    echo "scored_one_hold=true"
} >"$OUT/lock-window.txt"
echo "CAMPAIGN_PASS ts=$lock_end"

flock -u 9
exec 9>&-
lock_acquired=0
flock -n "$LOCK" -c 'echo GPU_LOCK_FREE_AFTER_CAMPAIGN'
assert_idle
assert_port_clear
trap - EXIT INT TERM
echo "CAMPAIGN_CLEAN_EXIT ts=$(date -u +%FT%TZ)"
