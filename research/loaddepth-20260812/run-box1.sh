#!/usr/bin/env bash
# Box1 deep-load curve: exactness c=12..24, then N=3 interleaved naked-default windows.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the exact box1 checkout commit}"
: "${EXPECTED_SERVER_SHA256:?set EXPECTED_SERVER_SHA256 to the staged release binary hash}"
REPO=${LOADDEPTH_REPO:-/home/ubuntu/memra-dualpp-flip}
ROOT=${LOADDEPTH_OUT:-/home/ubuntu/cx-loaddepth-20260812/raw}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${LOADDEPTH_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf
GOLDEN=/home/ubuntu/darktrain2/golden-response.bin
EXPECTED_GOLDEN=21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de
SERVER=$REPO/target/release/memra-server
QOS=$REPO/research/p0iso-20260810/qos_probe.py
LOAD=$SCRIPT_DIR/load_probe.py
REDUCE=$SCRIPT_DIR/reduce.py
PORT=${LOADDEPTH_PORT:-18524}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37

test ! -e "$ROOT" || { echo "FAIL: output already exists: $ROOT"; exit 1; }
mkdir -p "$ROOT"
exec > >(tee "$ROOT/driver.log") 2>&1

echo "LOCK_QUEUE_CHECK $(date -u +%FT%TZ)"
fuser -v /tmp/memra-gpu.lock 2>&1 || true
exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "LOADDEPTH_LOCK_ACQUIRED $(date -u +%FT%TZ) pid=$$ ppid=$PPID sid=$(ps -o sid= -p $$ | tr -d ' ')"

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
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free \
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
    echo "FAIL: server $pid did not stop"
    return 1
}

cleanup() {
    stop_server || true
    stop_sampler
}
trap cleanup EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 300); do
        curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "FAIL: server died during boot"
            tail -100 "$log"
            return 1
        fi
        sleep 2
    done
    echo "FAIL: server never became ready"
    tail -100 "$log"
    return 1
}

assert_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server fatal|illegal memory access|ILLEGAL_ADDRESS|sentinel|same boundary slot|mismatches=[1-9]' \
        "$log" \
       || grep -En 'MISMATCH' "$log"; then
        return 1
    fi
}

assert_dual_metrics() {
    python3 - "$1" <<'PY'
import json
import pathlib
import sys

metrics = json.loads(pathlib.Path(sys.argv[1]).read_text())["dual_pp"]
pairs = int(metrics["slot_pairs"])
uses = [int(value) for value in metrics["slot_uses"]]
assert pairs > 0, metrics
assert uses == [pairs, pairs], metrics
assert int(metrics["slot_collisions"]) == 0, metrics
PY
}

start_server() {
    local label=$1 log=$2
    echo "SERVER_START label=$label ts=$(date -u +%FT%TZ)"
    # This is the merged naked default: both dual policy variables are absent.
    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_DUAL_PP_TIMING -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_BG_JOB \
        -u MEMRA_DECODE_BATCH_CAP \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_MAX_SESSIONS=64 MEMRA_LANE_MAX_JUDGE=64 MEMRA_LANE_MAX_HARVEST=64 \
        MEMRA_SLO_P99_MS=1000000 MEMRA_TAG="cx-loaddepth-$label" \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
}

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
git merge-base --is-ancestor e94699eba HEAD
git diff --quiet
git diff --cached --quiet
for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" "$LOAD" "$REDUCE"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
test "$(sha256sum "$GOLDEN" | awk '{print $1}')" = "$EXPECTED_GOLDEN"
test "$(sha256sum "$SERVER" | awk '{print $1}')" = "$EXPECTED_SERVER_SHA256"
{
    echo "source_commit=$EXPECTED_SOURCE"
    echo "source_branch=$(git branch --show-current)"
    echo "flip_ancestor=e94699eba"
    echo "server_sha256=$EXPECTED_SERVER_SHA256"
    echo "protocol=naked Step-3.7 PP-2; no MEMRA_DUAL_PP/MEMRA_PP_OVERLAP"
    git log -5 --oneline --decorate
    echo "post_flip_changed_paths:"
    git diff --name-only e94699eba..HEAD
} >"$ROOT/provenance.txt"
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" "$LOAD" "$REDUCE" \
    >"$ROOT/SHA256SUMS"
snapshot "$ROOT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$ROOT/gpu.csv" 2>&1 &
sampler_pid=$!

# Exactness is strictly first and stops at the first bad new width.
mkdir -p "$ROOT/exactness"
exact_log=$ROOT/exactness/server.log
start_server exactness "$exact_log"
passed_widths=()
failed_width=
for width in 12 16 20 24; do
    dir=$ROOT/exactness/c$width
    mkdir -p "$dir"
    echo "EXACTNESS_START c=$width ts=$(date -u +%FT%TZ)"
    set +e
    python3 "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "cx-loaddepth-c$width" \
        --requests "$width" --max-tokens 64 --golden "$GOLDEN" \
        --lanes interactive,judge,harvest \
        --rows "$dir/qos-rows.jsonl" --summary "$dir/qos-summary.json" \
        2>&1 | tee "$dir/qos.log"
    qos_rc=${PIPESTATUS[0]}
    set -e
    echo "$qos_rc" >"$dir/qos.exit"
    if [[ $qos_rc -ne 0 ]]; then
        failed_width=$width
        echo "EXACTNESS_STOP c=$width rc=$qos_rc"
        break
    fi
    grep -q '"exactness": "match"' "$dir/qos-summary.json"
    grep -q "\"golden_matches\": $width" "$dir/qos-summary.json"
    passed_widths+=("$width")
    echo "EXACTNESS_PASS c=$width ts=$(date -u +%FT%TZ)"
done
set +e
curl -sf "$BASE/metrics" >"$ROOT/exactness/metrics-final.json"
exact_metrics_rc=$?
set -e
stop_server
set +e
assert_clean "$exact_log"
exact_server_clean_rc=$?
if [[ $exact_metrics_rc -eq 0 ]]; then
    assert_dual_metrics "$ROOT/exactness/metrics-final.json"
    exact_dual_metrics_rc=$?
else
    exact_dual_metrics_rc=1
fi
set -e
echo "$exact_metrics_rc" >"$ROOT/exactness/metrics.exit"
echo "$exact_server_clean_rc" >"$ROOT/exactness/server-clean.exit"
echo "$exact_dual_metrics_rc" >"$ROOT/exactness/dual-metrics.exit"
python3 - "$ROOT/exactness-verdict.json" "$failed_width" "${passed_widths[*]}" \
    "$exact_metrics_rc" "$exact_server_clean_rc" "$exact_dual_metrics_rc" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
failed = int(sys.argv[2]) if sys.argv[2] else None
passed = [int(value) for value in sys.argv[3].split()]
metrics_ok = int(sys.argv[4]) == 0
server_log_clean = int(sys.argv[5]) == 0
dual_metrics_ok = int(sys.argv[6]) == 0
path.write_text(json.dumps({
    "requested_widths": [12, 16, 20, 24],
    "passed_widths": passed,
    "failed_width": failed,
    "metrics_ok": metrics_ok,
    "server_log_clean": server_log_clean,
    "dual_metrics_ok": dual_metrics_ok,
    "expected_sha256": "21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de",
    "verdict": "PASS" if failed is None and metrics_ok and server_log_clean and dual_metrics_ok else "STOP_AT_FAILURE",
}, indent=2, sort_keys=True) + "\n")
PY
if [[ $exact_metrics_rc -ne 0 || $exact_server_clean_rc -ne 0 || $exact_dual_metrics_rc -ne 0 ]]; then
    echo "FAIL: exactness server or metrics error; verdict recorded"
    exit 1
fi

allowed_widths=(8 10)
for width in "${passed_widths[@]}"; do
    if [[ $width -ne 24 ]]; then
        allowed_widths+=("$width")
    fi
done
if [[ -z $failed_width ]]; then
    allowed_widths=(8 10 12 16 20 24)
fi

mkdir -p "$ROOT/perf"
orders=(
    "8 10 12 16 20 24"
    "24 20 16 12 10 8"
    "12 16 20 24 8 10"
)
for rep in 1 2 3; do
    read -r -a order <<<"${orders[$((rep - 1))]}"
    position=0
    for width in "${order[@]}"; do
        keep=0
        for allowed in "${allowed_widths[@]}"; do
            [[ $width -eq $allowed ]] && keep=1
        done
        [[ $keep -eq 1 ]] || continue
        position=$((position + 1))
        label=$(printf 'r%d-p%02d-c%02d' "$rep" "$position" "$width")
        dir=$ROOT/perf/$label
        mkdir -p "$dir"
        snapshot "$dir/thermal-before.log" "$label-before"
        start_server "$label" "$dir/server.log"
        python3 "$LOAD" --base "$BASE" --model "$MODEL_NAME" \
            --label "$label-warmup" --concurrency "$width" --max-tokens 128 \
            --out "$dir/warmup.jsonl" 2>&1 | tee "$dir/warmup.log"
        echo "SCORE_START label=$label ts=$(date -u +%FT%TZ)"
        python3 "$LOAD" --base "$BASE" --model "$MODEL_NAME" \
            --label "$label" --concurrency "$width" --max-tokens 128 \
            --out "$dir/score.jsonl" 2>&1 | tee "$dir/score.log"
        curl -sf "$BASE/metrics" >"$dir/metrics-final.json"
        stop_server
        assert_clean "$dir/server.log"
        assert_dual_metrics "$dir/metrics-final.json"
        grep -q 'decode wave cap 8; scheduler tick cap 16' "$dir/server.log"
        grep -q '\[dual-pp\] dual-active PP-2 decode engaged' "$dir/server.log"
        snapshot "$dir/thermal-after.log" "$label-after"
        echo "SCORE_PASS label=$label ts=$(date -u +%FT%TZ)"
    done
done

stop_sampler
snapshot "$ROOT/nvidia-smi-after.log" campaign-complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
python3 "$REDUCE" "$ROOT" --source "$EXPECTED_SOURCE" 2>&1 | tee "$ROOT/reduce.log"
echo "LOADDEPTH_ALL_PASS $(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
