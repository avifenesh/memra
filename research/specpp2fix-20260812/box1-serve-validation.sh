#!/usr/bin/env bash
# Spec-on golden matrix plus the three-boot c=8 sticky-crash soak on box1.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"

REPO=${SPEC_PP2FIX_REPO:-/home/ubuntu/memra-cx-specpp2fix}
OUT=${SPEC_PP2FIX_SERVE_OUT:-$REPO/research/specpp2fix-20260812/raw/box1/serve-validation}
MODEL_ROOT=${SPEC_PP2FIX_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${SPEC_PP2FIX_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${SPEC_PP2FIX_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
GOLDEN=${SPEC_PP2FIX_GOLDEN:-/home/ubuntu/darktrain2/golden-response.bin}
EXPECTED_GOLDEN=21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de
SERVER=$REPO/target/release/memra-server
QOS=$REPO/research/p0iso-20260810/qos_probe.py
LOAD=$REPO/tools/load-serve.py
PORT=${SPEC_PP2FIX_SERVE_PORT:-18482}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37
WIDTHS=(1 2 4 8 16)

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1
started=$(date -u +%FT%TZ)

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

assert_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server fatal|illegal memory access|ILLEGAL_ADDRESS|sentinel|same boundary slot|mismatches=[1-9]' \
        "$log" || grep -En 'MISMATCH' "$log"; then
        return 1
    fi
}

start_server() {
    local arm=$1 label=$2 log=$3
    local -a policy=()
    if [[ $arm == serial ]]; then
        policy=(MEMRA_DUAL_PP=0)
    fi
    env \
        -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP \
        -u MEMRA_SPEC_PIPE \
        -u MEMRA_SPEC_NOGRAPH \
        "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_SERVE_SPEC=1 \
        MEMRA_SPEC_GATE=0 \
        MEMRA_SPEC_K=1 \
        MEMRA_SPEC_STATS=1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_MAX_SESSIONS=64 \
        MEMRA_LANE_MAX_JUDGE=64 \
        MEMRA_LANE_MAX_HARVEST=64 \
        MEMRA_SLO_P99_MS=1000000 \
        MEMRA_TAG="specpp2fix-$label" \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
}

finish_server() {
    local arm=$1 log=$2
    stop_server
    assert_clean "$log"
    test "$(grep -c '\[spec-acc\]' "$log")" -gt 0
    if [[ $arm == dual ]]; then
        grep -q 'decode wave cap 8; scheduler tick cap 16 (dual PP, default-off arm)' "$log"
    else
        grep -q 'decode wave cap 8; scheduler tick cap 8' "$log"
        if grep -q '\[dual-pp\]' "$log"; then
            echo "FAIL: dual marker present in serial server $log"
            return 1
        fi
    fi
}

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" "$LOAD"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
test "$(sha256sum "$GOLDEN" | awk '{print $1}')" = "$EXPECTED_GOLDEN"
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" "$LOAD" >"$OUT/SHA256SUMS"

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "SERVE_VALIDATION_LOCK_ACQUIRED $(date -u +%FT%TZ)"
snapshot "$OUT/nvidia-smi-before.log" serve-validation-start
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu.csv" 2>&1 &
sampler_pid=$!

# One fresh process per policy arm; every point is a barrier release at the named concurrency.
for arm in dual serial; do
    label=golden-$arm
    server_log=$OUT/$label-server.log
    echo "golden_arm_start=$arm ts=$(date -u +%FT%TZ)"
    start_server "$arm" "$label" "$server_log"
    curl -sf "$BASE/metrics" >"$OUT/$label-metrics-before.json"
    for width in "${WIDTHS[@]}"; do
        point=$(printf 'c%02d' "$width")
        dir=$OUT/$label/$point
        mkdir -p "$dir"
        "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "$label-$point" \
            --requests "$width" --max-tokens 64 --golden "$GOLDEN" \
            --lanes interactive,judge,harvest \
            --rows "$dir/qos-rows.jsonl" --summary "$dir/qos-summary.json" \
            2>&1 | tee "$dir/qos.log"
        grep -q '"exactness": "match"' "$dir/qos-summary.json"
        grep -q "\"golden_matches\": $width" "$dir/qos-summary.json"
        echo "golden_point=$label-$point result=PASS"
    done
    curl -sf "$BASE/metrics" >"$OUT/$label-metrics-after.json"
    finish_server "$arm" "$server_log"
    echo "golden_arm_done=$arm ts=$(date -u +%FT%TZ)"
done

# Three independent CUDA contexts, each with 64 measured requests held at c=8.
for boot in 1 2 3; do
    label=$(printf 'soak-boot-%02d-dual-c8' "$boot")
    dir=$OUT/$label
    mkdir -p "$dir"
    server_log=$dir/server.log
    echo "soak_boot_start=$label ts=$(date -u +%FT%TZ)"
    snapshot "$dir/thermal-before.log" "$label-before"
    start_server dual "$label" "$server_log"
    curl -sf "$BASE/metrics" >"$dir/metrics-before.json"
    python3 "$LOAD" --base "$BASE" --model "$MODEL_NAME" \
        --concurrency 8 --requests 64 --max-tokens 64 --greedy --warmup 0 \
        --label "$label" --out "$dir/points.jsonl" --per-request "$dir/requests.jsonl" \
        2>&1 | tee "$dir/load.log"
    python3 - "$dir/points.jsonl" "$label" <<'PY'
import json
import pathlib
import sys

rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
assert len(rows) == 1, rows
row = rows[0]
assert row["label"] == sys.argv[2], row
assert row["n_ok"] == 64 and row["n_err"] == 0 and row["n_shed"] == 0, row
assert row["completion_tokens_total"] == 4096, row
PY
    curl -sf "$BASE/metrics" >"$dir/metrics-after.json"
    finish_server dual "$server_log"
    snapshot "$dir/thermal-after.log" "$label-after"
    echo "soak_boot_done=$label ts=$(date -u +%FT%TZ)"
done

stop_sampler
journalctl -k --since "$started" --no-pager >"$OUT/kernel-since-start.log" 2>&1 || true
grep -Ein \
    'CUDA_ERROR|illegal memory access|ILLEGAL_ADDRESS|sentinel|panicked at|worker.*died|server fatal|mismatches=[1-9]' \
    "$OUT"/*-server.log "$OUT"/soak-boot-*/server.log >"$OUT/failure-scan.log" || true
test ! -s "$OUT/failure-scan.log" || { cat "$OUT/failure-scan.log"; exit 1; }

python3 - "$OUT" "$EXPECTED_GOLDEN" "$EXPECTED_SOURCE" <<'PY' | tee "$OUT/reduce.log"
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
source = sys.argv[3]

golden_paths = sorted(root.glob("golden-*/c*/qos-summary.json"))
assert len(golden_paths) == 10, golden_paths
golden = [json.loads(path.read_text()) for path in golden_paths]
assert all(row["exactness"] == "match" for row in golden)
assert all(row["expected_sha256"] == expected for row in golden)
assert all(row["golden_matches"] == row["requests"] for row in golden)
assert all(row["hash_counts"] == {expected: row["requests"]} for row in golden)
for arm in ("dual", "serial"):
    widths = sorted(row["requests"] for row in golden if f"golden-{arm}" in row["label"])
    assert widths == [1, 2, 4, 8, 16], (arm, widths)

soak_paths = sorted(root.glob("soak-boot-*/points.jsonl"))
assert len(soak_paths) == 3, soak_paths
soak = [json.loads(path.read_text().strip()) for path in soak_paths]
assert all(row["n_ok"] == 64 and row["n_err"] == 0 and row["n_shed"] == 0 for row in soak)
assert all(row["completion_tokens_total"] == 4096 for row in soak)

temperatures = []
clocks = []
with (root / "gpu.csv").open(newline="", errors="replace") as stream:
    for row in csv.reader(stream):
        if len(row) < 7:
            continue
        try:
            temperatures.append(float(row[3].strip()))
            clocks.append(float(row[6].strip()))
        except ValueError:
            pass
assert temperatures and clocks

summary = {
    "schema": "memra.specpp2fix.serve-validation.v1",
    "source_commit": source,
    "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition",
    "spec_policy": {"serve_spec": 1, "spec_gate": 0, "k": 1},
    "golden_matrix": {
        "arms": ["dual-naked", "serial"],
        "concurrency": [1, 2, 4, 8, 16],
        "points": len(golden),
        "requests": sum(row["requests"] for row in golden),
        "golden_matches": sum(row["golden_matches"] for row in golden),
        "expected_sha256": expected,
        "verdict": "PASS",
    },
    "sticky_crash_soak": {
        "fresh_boots": len(soak),
        "concurrency": 8,
        "requests_per_boot": 64,
        "max_tokens": 64,
        "requests_ok": sum(row["n_ok"] for row in soak),
        "errors": sum(row["n_err"] for row in soak),
        "completion_tokens": sum(row["completion_tokens_total"] for row in soak),
        "aggregate_tok_s": [row["agg_tok_s"] for row in soak],
        "verdict": "PASS",
    },
    "thermal_regime": {
        "artificial_cooldown": False,
        "sample_interval_ms": 250,
        "samples": len(temperatures),
        "temperature_c_min": min(temperatures),
        "temperature_c_max": max(temperatures),
        "sm_clock_mhz_min": min(clocks),
        "sm_clock_mhz_max": max(clocks),
    },
    "verdict": "PASS",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

snapshot "$OUT/nvidia-smi-after.log" serve-validation-complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "SERVE_VALIDATION_LOCK_RELEASED $(date -u +%FT%TZ)"
flock -u 9
trap - EXIT INT TERM
echo "SPEC_PP2FIX_SERVE_VALIDATION_PASS $(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
