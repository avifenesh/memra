#!/usr/bin/env bash
# N=5 interleaved c=8 forced-spec K=1 versus spec-off timing on naked dual-active PP-2.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"

REPO=${SPEC_PP2FIX_REPO:-/home/ubuntu/memra-cx-specpp2fix}
OUT=${SPEC_PP2FIX_TIMING_OUT:-$REPO/research/specpp2fix-20260812/raw/box1/timing}
MODEL_ROOT=${SPEC_PP2FIX_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${SPEC_PP2FIX_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${SPEC_PP2FIX_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
SERVER=$REPO/target/release/memra-server
LOAD=$REPO/tools/load-serve.py
PORT=${SPEC_PP2FIX_TIMING_PORT:-18483}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37

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

run_arm() {
    local rep=$1 arm=$2
    local label=r${rep}-${arm}
    local server_log=$OUT/$label-server.log
    local -a policy

    case "$arm" in
        spec-off) policy=(MEMRA_SERVE_SPEC=0) ;;
        spec-on) policy=(MEMRA_SERVE_SPEC=1 MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1) ;;
        *) echo "FAIL: unknown arm $arm"; return 1 ;;
    esac

    echo "timing_cell_start=$label ts=$(date -u +%FT%TZ)"
    snapshot "$OUT/$label-thermal-before.log" "$label-before"
    env \
        -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_GATE \
        -u MEMRA_SPEC_K \
        -u MEMRA_SPEC_STATS \
        -u MEMRA_SPEC_PIPE \
        -u MEMRA_SPEC_NOGRAPH \
        "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai \
        MEMRA_ADDR="127.0.0.1:$PORT" \
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
        MEMRA_TAG="specpp2fix-timing-$label" \
        "$SERVER" >"$server_log" 2>&1 &
    server_pid=$!
    wait_ready "$server_log"
    curl -sf "$BASE/metrics" >"$OUT/$label-metrics-before.json"

    python3 "$LOAD" --base "$BASE" --model "$MODEL_NAME" \
        --concurrency 8 --requests 32 --max-tokens 128 --greedy --warmup 1 \
        --label "$label" --out "$OUT/points.jsonl" --per-request "$OUT/$label-requests.jsonl" \
        2>&1 | tee "$OUT/$label-load.log"
    curl -sf "$BASE/metrics" >"$OUT/$label-metrics-after.json"
    stop_server

    if grep -Ein \
        'CUDA_ERROR|out of memory|panicked at|worker.*died|server fatal|illegal memory access|ILLEGAL_ADDRESS|sentinel|mismatches=[1-9]' \
        "$server_log" >"$OUT/$label-failure-scan.log"; then
        cat "$OUT/$label-failure-scan.log"
        return 1
    fi
    if [[ $arm == spec-on ]]; then
        test "$(grep -c '\[spec-acc\]' "$server_log")" -gt 0
    elif grep -q '\[spec-acc\]' "$server_log"; then
        echo "FAIL: spec-off arm emitted spec acceptance lines"
        return 1
    fi
    grep -q 'decode wave cap 8; scheduler tick cap 16 (dual PP, default-off arm)' "$server_log"
    snapshot "$OUT/$label-thermal-after.log" "$label-after"
    echo "timing_cell_done=$label ts=$(date -u +%FT%TZ)"
}

for artifact in "$MODEL" "$DRAFT" "$SERVER" "$LOAD"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
sha256sum "$MODEL" "$DRAFT" "$SERVER" "$LOAD" >"$OUT/SHA256SUMS"

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "TIMING_LOCK_ACQUIRED $(date -u +%FT%TZ)"
snapshot "$OUT/nvidia-smi-before.log" timing-start
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu.csv" 2>&1 &
sampler_pid=$!

# Alternate pair order to distribute both arms through the thermal window.
run_arm 1 spec-off
run_arm 1 spec-on
run_arm 2 spec-on
run_arm 2 spec-off
run_arm 3 spec-off
run_arm 3 spec-on
run_arm 4 spec-on
run_arm 4 spec-off
run_arm 5 spec-off
run_arm 5 spec-on

stop_sampler
journalctl -k --since "$started" --no-pager >"$OUT/kernel-since-start.log" 2>&1 || true

python3 - "$OUT" "$EXPECTED_SOURCE" <<'PY' | tee "$OUT/reduce.log"
import csv
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
source = sys.argv[2]
rows = [json.loads(line) for line in (root / "points.jsonl").read_text().splitlines()]
assert len(rows) == 10, len(rows)
assert all(row["concurrency"] == 8 for row in rows)
assert all(row["requests"] == 32 for row in rows)
assert all(row["n_ok"] == 32 and row["n_err"] == 0 and row["n_shed"] == 0 for row in rows)
assert all(row["completion_tokens_total"] == 4096 for row in rows)

by_arm = {
    arm: sorted((row for row in rows if row["label"].endswith(arm)), key=lambda row: row["label"])
    for arm in ("spec-off", "spec-on")
}
assert all(len(arm_rows) == 5 for arm_rows in by_arm.values()), by_arm

accept_pattern = re.compile(r"\[spec-acc\].*?burst=(\d+)/(\d+)")
accept_by_rep = []
total_accepted = 0
total_drafted = 0
for path in sorted(root.glob("r*-spec-on-server.log")):
    pairs = [(int(a), int(d)) for a, d in accept_pattern.findall(path.read_text(errors="replace"))]
    assert pairs, path
    accepted = sum(a for a, _ in pairs)
    drafted = sum(d for _, d in pairs)
    total_accepted += accepted
    total_drafted += drafted
    accept_by_rep.append({
        "label": path.name.removesuffix("-server.log"),
        "accepted": accepted,
        "drafted": drafted,
        "rate": accepted / drafted,
    })

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

arm_summary = {}
for arm, arm_rows in by_arm.items():
    values = [row["agg_tok_s"] for row in arm_rows]
    arm_summary[arm] = {
        "N": len(values),
        "agg_tok_s_by_rep": values,
        "agg_tok_s_median": statistics.median(values),
        "agg_tok_s_min": min(values),
        "agg_tok_s_max": max(values),
    }

off = arm_summary["spec-off"]["agg_tok_s_median"]
on = arm_summary["spec-on"]["agg_tok_s_median"]
summary = {
    "schema": "memra.specpp2fix.timing.v1",
    "source_commit": source,
    "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition",
    "protocol": "N=5 per arm, alternating pair order, c=8, 32x128-token measured requests per cell, one GPU-lock hold",
    "arms": arm_summary,
    "spec_acceptance": {
        "k": 1,
        "accepted": total_accepted,
        "drafted": total_drafted,
        "rate": total_accepted / total_drafted,
        "by_rep": accept_by_rep,
    },
    "spec_on_over_off": on / off,
    "spec_on_percent_change": (on / off - 1.0) * 100.0,
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

snapshot "$OUT/nvidia-smi-after.log" timing-complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "TIMING_LOCK_RELEASED $(date -u +%FT%TZ)"
flock -u 9
trap - EXIT INT TERM
echo "SPEC_PP2FIX_TIMING_PASS $(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
