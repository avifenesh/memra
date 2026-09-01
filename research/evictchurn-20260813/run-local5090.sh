#!/usr/bin/env bash
# Prefix-cache eviction contention battery for the thermally capped local RTX 5090.
# This is a behavior receipt. It never changes clocks and makes no throughput claim.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
LANE=$ROOT/research/evictchurn-20260813
MODEL=${EVICTCHURN_MODEL:-/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
MODEL_NAME=${EVICTCHURN_MODEL_NAME:-q36-35b}
PORT=${EVICTCHURN_PORT:-18514}
BASE=http://127.0.0.1:$PORT
STAMP=${EVICTCHURN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${EVICTCHURN_OUT:-$LANE/raw/run-$STAMP}
SERVER=$ROOT/target/release/memra-server
WORKING_SET=40
TENANTS=4
TARGET_ENTRIES=12
PREFIX_TOKENS=256
SUFFIX_TOKENS=8
MAX_TOKENS=8
SERVER_PID=
SAMPLER_PID=

test -f "$MODEL"
test -x "$SERVER"
test -f "$LANE/evict_churn.py"
test -f "$ROOT/research/prefixmoney-20260812/prefix_gate.py"
test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

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
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu \
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
    local server_log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "FAIL: server died before readiness"
            tail -200 "$server_log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server readiness timeout"
    tail -200 "$server_log"
    return 1
}

start_server() {
    local label=$1 cache_mb=$2
    local server_log=$OUT/server-$label.log
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SWA_RING \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="$MODEL_NAME=$MODEL" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_PREFIX_CACHE_MB="$cache_mb" MEMRA_PREFIX_DEDUP=1 \
        MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 MEMRA_SERVE_SPEC=0 \
        MEMRA_CTX=512 MEMRA_MAX_SESSIONS=4 \
        nice -n 10 ionice -c 2 -n 7 "$SERVER" >"$server_log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$server_log"
    curl -sf "$BASE/metrics" >"$OUT/metrics-$label-before.json"
}

finish_server() {
    local label=$1
    curl -sf "$BASE/metrics" >"$OUT/metrics-$label-after.json"
    stop_server
}

run_harness() {
    local label=$1 pattern=$2 requests=$3 cache_mb=$4
    start_server "$label" "$cache_mb"
    set +e
    nice -n 10 ionice -c 2 -n 7 timeout 7200 \
        python3 "$LANE/evict_churn.py" \
        --base "$BASE" --model "$MODEL_NAME" \
        --out "$OUT/$label.jsonl" --server-log "$OUT/server-$label.log" \
        --namespace "evictchurn-$label" --pattern "$pattern" \
        --working-set "$WORKING_SET" --tenants "$TENANTS" --requests "$requests" \
        --prefix-tokens "$PREFIX_TOKENS" --suffix-tokens "$SUFFIX_TOKENS" \
        --max-tokens "$MAX_TOKENS" \
        2>&1 | tee "$OUT/$label.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/$label.exit"
    finish_server "$label"
    test "$rc" -eq 0 || { echo "FAIL: $label harness rc=$rc"; return "$rc"; }
}

finalize() {
    stop_server
    stop_sampler
    snapshot "$OUT/nvidia-smi-after.log" complete
    rg -n -i 'prefix-cache|refused|CUDA_ERROR|out of memory|panicked' \
        "$OUT"/server-*.log >"$OUT/server-markers.log" || true
    awk -F, '{gsub(/ /, "", $6); if ($6+0 > max) max=$6+0} END {print "max_observed_sm_clock_mhz=" max}' \
        "$OUT/gpu.csv" >"$OUT/thermal-summary.txt"
    find "$OUT" -maxdepth 1 -type f ! -name SHA256SUMS ! -name driver.log -print0 \
        | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS"
}

run_locked() {
    local recent_battery
    recent_battery=$(find /tmp -maxdepth 1 -type f -name 'battery-*.log' -mmin -5 -print)
    test -z "$recent_battery" || {
        echo "$recent_battery"
        echo "FAIL: a battery log changed in the last five minutes"
        return 75
    }
    local apps
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: local 5090 is not idle"; return 75; }

    snapshot "$OUT/nvidia-smi-before.log" start
    sha256sum "$MODEL" "$SERVER" "$LANE/evict_churn.py" \
        "$ROOT/research/prefixmoney-20260812/prefix_gate.py" >"$OUT/SHA256SUMS.input"
    {
        echo "server_source_commit=$(git rev-parse HEAD)"
        echo "base_tag=$(git describe --tags --exact-match 18885ec479d897a3e8c42b0d408a71fa3edaa708)"
        echo "branch=$(git branch --show-current)"
        echo "model=$MODEL"
        echo "model_name=$MODEL_NAME"
        echo "rig=local RTX 5090 Laptop GPU"
        echo "thermal_regime=global 210-1200 MHz cap; no clock changes"
        echo "shape=single-GPU plain batched serving; spec off; reuse/affinity pools off"
        echo "working_set=$WORKING_SET"
        echo "tenants=$TENANTS"
        echo "target_equal_size_entries=$TARGET_ENTRIES"
        echo "prefix_tokens=$PREFIX_TOKENS"
        echo "suffix_tokens=$SUFFIX_TOKENS"
        echo "max_tokens=$MAX_TOKENS"
        echo "round_robin_requests=80"
        echo "hotset_requests=160 (80% to 20% prefixes; alpha=1.0; seed=3407)"
        echo "sequential_scan_requests=40 (every request is a new prefix)"
    } >"$OUT/provenance.txt"

    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/gpu.csv" 2>&1 &
    SAMPLER_PID=$!

    # One real entry under a deliberately roomy budget gives the byte denominator. The fixed
    # scored budget is then selected once and reused unchanged for all three isolated runs.
    run_harness calibration sequential-scan 1 256
    local entry_bytes cache_mb equal_entry_capacity hot_entries
    entry_bytes=$(python3 - "$OUT/calibration.jsonl" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
summary = next(row for row in rows if row.get("kind") == "summary")
print(summary["resident_bytes_final"])
PY
    )
    test "$entry_bytes" -gt 0 || { echo "FAIL: calibration inserted no resident bytes"; return 1; }
    cache_mb=$(python3 - "$entry_bytes" "$TARGET_ENTRIES" <<'PY'
import math
import sys
print(math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / (1024 * 1024)))
PY
    )
    equal_entry_capacity=$((cache_mb * 1024 * 1024 / entry_bytes))
    hot_entries=$((WORKING_SET / 5))
    test "$equal_entry_capacity" -ge "$hot_entries" || {
        echo "FAIL: calibrated budget cannot hold the hot subset"
        return 1
    }
    test "$equal_entry_capacity" -lt "$WORKING_SET" || {
        echo "FAIL: calibrated budget holds the whole working set"
        return 1
    }
    {
        echo "calibration_entry_bytes=$entry_bytes"
        echo "fixed_cache_mb=$cache_mb"
        echo "fixed_cache_bytes=$((cache_mb * 1024 * 1024))"
        echo "equal_entry_capacity=$equal_entry_capacity"
        echo "working_set_fraction=$(python3 -c "print($equal_entry_capacity / $WORKING_SET)")"
    } | tee "$OUT/cache-budget.txt"

    start_server exactness "$cache_mb"
    set +e
    nice -n 10 ionice -c 2 -n 7 timeout 7200 \
        python3 "$ROOT/research/prefixmoney-20260812/prefix_gate.py" \
        --base "$BASE" --model "$MODEL_NAME" --out "$OUT/exactness.jsonl" \
        --namespace evictchurn-exact --reps 3 --prefix-tokens "$PREFIX_TOKENS" \
        --suffix-tokens 16 --max-tokens "$MAX_TOKENS" --concurrency 2 \
        2>&1 | tee "$OUT/exactness.log"
    local exact_rc=${PIPESTATUS[0]}
    set -e
    echo "$exact_rc" >"$OUT/exactness.exit"
    finish_server exactness
    test "$exact_rc" -eq 0 || { echo "FAIL: prefix exactness rc=$exact_rc"; return "$exact_rc"; }

    run_harness round-robin round-robin 80 "$cache_mb"
    run_harness hotset hotset 160 "$cache_mb"
    run_harness sequential-scan sequential-scan 40 "$cache_mb"

    finalize
    apps=$(compute_apps)
    test -z "$apps" || { echo "$apps"; echo "FAIL: GPU process remained"; return 1; }
    echo "EVICTCHURN_LOCAL_PASS $(date -u +%FT%TZ)"
}

(
    flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ)"
    set +e
    run_locked
    rc=$?
    set -e
    echo "lock_release=$(date -u +%FT%TZ)"
    exit "$rc"
) 9>/tmp/gpu5090.lock
