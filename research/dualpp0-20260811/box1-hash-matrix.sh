#!/usr/bin/env bash
# Fresh-boot 21b8293f golden plus c=1..8 per-request one-hash matrix, serial and dual.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp0}
OUT=${DUALPP_HASH_OUT:-$REPO/research/dualpp0-20260811/raw/box1/hash-matrix}
MODEL_ROOT=${DUALPP_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${DUALPP_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DUALPP_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
GOLDEN=${DUALPP_GOLDEN:-/home/ubuntu/darktrain2/golden-response.bin}
EXPECTED_GOLDEN=21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de
SERVER=$REPO/target/release/memra-server
QOS=$REPO/research/p0iso-20260810/qos_probe.py
PORT=${DUALPP_HASH_PORT:-18458}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=

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

start_server() {
    local arm=$1 label=$2 log=$3
    local -a policy=(MEMRA_DUAL_PP=0)
    if [[ $arm == dual ]]; then
        policy=(MEMRA_DUAL_PP=1 MEMRA_PP_OVERLAP=1)
    fi
    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_SPEC_K \
        -u MEMRA_SERVE_BATCH -u MEMRA_BG_JOB "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_TAG="dualpp-$label" "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
}

assert_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server fatal|illegal|sentinel' \
        "$log"; then
        return 1
    fi
}

run_qos() {
    local arm=$1 label=$2 width=$3
    local dir=$OUT/$label
    mkdir -p "$dir"
    "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "$label" \
        --requests "$width" --max-tokens 64 --golden "$GOLDEN" \
        --rows "$dir/qos-rows.jsonl" --summary "$dir/qos-summary.json" \
        2>&1 | tee "$dir/qos.log"
    grep -q '"exactness": "match"' "$dir/qos-summary.json"
    grep -q "\"golden_matches\": $width" "$dir/qos-summary.json"
    echo "hash_point=$label arm=$arm c=$width result=PASS"
}

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "HASH_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
test "$(sha256sum "$GOLDEN" | awk '{print $1}')" = "$EXPECTED_GOLDEN"
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

# Ten independent process boots, alternating the default-off arm and the explicitly enabled
# arm. c=1 must remain the honest serial fallback in both.
for boot in $(seq 1 10); do
    if (( boot % 2 == 1 )); then arm=serial; else arm=dual; fi
    label=$(printf 'boot-%02d-%s' "$boot" "$arm")
    log=$OUT/$label-server.log
    start_server "$arm" "$label" "$log"
    run_qos "$arm" "$label" 1
    stop_server
    assert_clean "$log"
    ! grep -q '\[dual-pp\]' "$log" # c=1 is serial even when the dual door is armed
done

# One loaded process per arm, every concurrency width. The gate compares each request to the
# same frozen 326-byte completion, not merely one aggregate hash for the batch.
for arm in serial dual; do
    label="matrix-$arm"
    log=$OUT/$label-server.log
    start_server "$arm" "$label" "$log"
    for width in $(seq 1 8); do
        run_qos "$arm" "$label-c$width" "$width"
    done
    stop_server
    assert_clean "$log"
    if [[ $arm == dual ]]; then
        grep -q '\[dual-pp\] dual-active PP-2 decode engaged' "$log"
    else
        ! grep -q '\[dual-pp\]' "$log"
    fi
done

python3 - "$OUT" "$EXPECTED_GOLDEN" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
summaries = sorted(root.glob("*/qos-summary.json"))
assert len(summaries) == 26, len(summaries)
requests = 0
for path in summaries:
    row = json.loads(path.read_text())
    assert row["exactness"] == "match", (path, row)
    assert row["expected_sha256"] == expected, (path, row)
    assert row["golden_matches"] == row["requests"], (path, row)
    assert row["hash_counts"] == {expected: row["requests"]}, (path, row)
    requests += row["requests"]
receipt = {
    "schema": "memra.dualpp0.hash-matrix.v1",
    "fresh_boots": 10,
    "matrix_widths": list(range(1, 9)),
    "arms": ["serial", "dual"],
    "summaries": len(summaries),
    "requests": requests,
    "expected_sha256": expected,
    "verdict": "PASS",
}
(root / "summary.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY

snapshot "$OUT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "HASH_MATRIX_PASS $(date -u +%FT%TZ)"
