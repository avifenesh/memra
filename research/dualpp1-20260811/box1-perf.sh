#!/usr/bin/env bash
# N=5 interleaved serial/dual arbitrary-width curve, c=2..17, 512-token decode windows.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
: "${DUALPP_LOCK_HELD:?run through box1-run.sh so fd 9 owns /tmp/memra-gpu.lock}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp1}
OUT=${DUALPP_PERF_OUT:-$REPO/research/dualpp1-20260811/raw/box1/perf}
MODEL_ROOT=${DUALPP_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${DUALPP_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
SERVER=$REPO/target/release/memra-server
BENCH=$REPO/research/newboxgates-20260811/serve_bench.py
PORT=${DUALPP_PERF_PORT:-18459}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37
WIDTH_ORDER=(8 16 2 17 3 15 4 14 5 13 6 12 7 11 9 10)

if ! test -e /proc/$$/fd/9 || ! flock -n 9; then
    echo "FAIL: inherited GPU lock missing"
    exit 75
fi
test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

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
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|illegal|sentinel|same boundary slot' \
        "$log"; then
        return 1
    fi
}

run_point() {
    local rep=$1 arm=$2 width=$3
    local label="r${rep}-${arm}-c${width}"
    echo "point_start=$label ts=$(date -u +%FT%TZ)"
    if [[ $arm == dual ]]; then
        curl -sf "$BASE/metrics" >"$OUT/$label-metrics-before.json"
    fi
    "$BENCH" --base "$BASE" --model "$MODEL_NAME" --shape decode --label "$label" \
        --concurrency "$width" --max-tokens 512 --require-length \
        --out "$OUT/points.jsonl" 2>&1 | tee "$OUT/$label-load.log"
    if [[ $arm == dual ]]; then
        curl -sf "$BASE/metrics" >"$OUT/$label-metrics-after.json"
    fi
    echo "point_done=$label ts=$(date -u +%FT%TZ)"
}

run_arm() {
    local rep=$1 arm=$2 offset=$3
    local label="r${rep}-${arm}"
    local log=$OUT/$label-server.log
    local -a policy=(MEMRA_DUAL_PP=0)
    if [[ $arm == dual ]]; then
        policy=(MEMRA_DUAL_PP=1 MEMRA_PP_OVERLAP=1)
    fi
    echo "arm_start=$label ts=$(date -u +%FT%TZ) offset=$offset"
    {
        echo "arm=$arm"
        echo "MEMRA_DUAL_PP=$([[ $arm == dual ]] && echo 1 || echo 0)"
        echo "MEMRA_PP_OVERLAP=$([[ $arm == dual ]] && echo 1 || echo '<unset>')"
        echo "MEMRA_DUAL_PP_TIMING=<unset>"
        echo "MEMRA_SERVE_SPEC=0"
        echo "MEMRA_PP_STAGES=2"
        echo "MEMRA_PP_DEVICES=0,1"
        echo "widths=2..17"
    } >"$OUT/$label-env.txt"
    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_DUAL_PP_TIMING \
        -u MEMRA_PP_HOST_BOUNCE -u MEMRA_SIG_ROUTER -u MEMRA_MOE_DEV \
        -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_BG_JOB \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_SWA_RING "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_MAX_SESSIONS=64 MEMRA_TAG="dualpp1-perf-$label" \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
    "$BENCH" --base "$BASE" --model "$MODEL_NAME" --shape warmup \
        --label "$label-warmup" --concurrency 1 --max-tokens 16 \
        --out "$OUT/warmups.jsonl" >"$OUT/$label-warmup.log" 2>&1
    snapshot "$OUT/$label-thermal-before.log" "$label-before"
    for point in $(seq 0 15); do
        index=$(( (point + offset) % ${#WIDTH_ORDER[@]} ))
        run_point "$rep" "$arm" "${WIDTH_ORDER[$index]}"
    done
    curl -sf "$BASE/metrics" >"$OUT/$label-metrics-final.json"
    stop_server
    assert_clean "$log"
    if [[ $arm == dual ]]; then
        grep -q 'decode wave cap 8; scheduler tick cap 16 (dual PP, default-off arm)' "$log"
        grep -q '\[dual-pp\] dual-active PP-2 decode engaged' "$log"
    else
        grep -q 'decode wave cap 8; scheduler tick cap 8' "$log"
        if grep -q '\[dual-pp\]' "$log"; then
            echo "FAIL: dual marker present in serial arm $label"
            return 1
        fi
    fi
    snapshot "$OUT/$label-thermal-after.log" "$label-after"
    echo "arm_done=$label ts=$(date -u +%FT%TZ)"
}

reduce() {
    python3 - "$OUT/points.jsonl" "$OUT" "$EXPECTED_SOURCE" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

points_path = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
source = sys.argv[3]
rows = [json.loads(line) for line in points_path.read_text().splitlines()]
requests = [row for row in rows if row.get("kind") == "request"]
points = [row for row in rows if row.get("kind") == "summary"]
assert len(points) == 160, len(points)
assert len(requests) == 1520, len(requests)
assert all(row["ok"] for row in requests), [row for row in requests if not row["ok"]]
assert all(point["n_error"] == 0 for point in points), points
assert all(point["completion_tokens_total"] == point["concurrency"] * 512 for point in points)
assert all(point["finish_reasons"] == ["length"] for point in points)

def series(arm, width):
    suffix = f"-{arm}-c{width}"
    return [float(row["decode_window_tok_s"]) for row in points if row["label"].endswith(suffix)]

widths = {}
for width in range(2, 18):
    serial = series("serial", width)
    dual = series("dual", width)
    assert len(serial) == len(dual) == 5, (width, serial, dual)
    sm = statistics.median(serial)
    dm = statistics.median(dual)
    widths[f"c{width}"] = {
        "N": 5,
        "serial_decode_window_tok_s": serial,
        "dual_decode_window_tok_s": dual,
        "serial_median_tok_s": sm,
        "dual_median_tok_s": dm,
        "delta_pct": (dm / sm - 1.0) * 100.0,
    }

slot_points = {}
for rep in range(1, 6):
    for width in range(2, 18):
        label = f"r{rep}-dual-c{width}"
        before = json.loads((out / f"{label}-metrics-before.json").read_text())
        after = json.loads((out / f"{label}-metrics-after.json").read_text())
        b = before.get("dual_pp", {})
        a = after["dual_pp"]
        pairs = int(a["slot_pairs"]) - int(b.get("slot_pairs", 0))
        uses = [
            int(a["slot_uses"][i]) - int((b.get("slot_uses") or [0, 0])[i])
            for i in range(2)
        ]
        collisions = int(a["slot_collisions"]) - int(b.get("slot_collisions", 0))
        overlaps = int(a["overlaps"]) - int(b.get("overlaps", 0))
        assert pairs > 0, (label, b, a)
        assert uses == [pairs, pairs], (label, uses, pairs)
        assert collisions == 0, (label, b, a)
        assert overlaps > 0, (label, b, a)
        slot_points.setdefault(f"c{width}", []).append({
            "rep": rep,
            "slot_pairs": pairs,
            "slot_uses": uses,
            "slot_collisions": collisions,
            "overlaps": overlaps,
        })

temperatures = []
clocks = []
with (out / "gpu.csv").open(newline="", errors="replace") as stream:
    for row in csv.reader(stream):
        if len(row) < 7:
            continue
        try:
            temperatures.append(float(row[3].strip()))
            clocks.append(float(row[6].strip()))
        except ValueError:
            pass
assert temperatures and clocks

floor_widths = {key: row["delta_pct"] for key, row in widths.items()
                if int(key[1:]) >= 8}
minimum_key = min(floor_widths, key=floor_widths.get)
minimum_delta = floor_widths[minimum_key]
verdict = "HOLD" if minimum_delta >= 15.0 else "REGRESSION"
summary = {
    "schema": "memra.dualpp1.perf.v1",
    "source_commit": source,
    "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition",
    "protocol": "N=5 interleaved serial/dual arms, rotated c2..17 order, 512 tokens/request, one inherited GPU lock hold",
    "metric": "aggregate completion tokens after first visible token / decode window second",
    "max_tokens_per_request": 512,
    "widths": widths,
    "dual_slot_deltas": slot_points,
    "thermal_regime": {
        "artificial_cooldown": False,
        "sample_interval_ms": 250,
        "samples": len(temperatures),
        "temperature_c_min": min(temperatures),
        "temperature_c_max": max(temperatures),
        "sm_clock_mhz_min": min(clocks),
        "sm_clock_mhz_max": max(clocks),
    },
    "hold_floor": {
        "threshold_pct_c_ge_8": 15.0,
        "minimum_width": minimum_key,
        "minimum_observed_pct": minimum_delta,
        "verdict": verdict,
    },
    "receipt": {"points": len(points), "request_rows": len(requests), "errors": 0},
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
}

for artifact in "$MODEL" "$SERVER" "$BENCH"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
sha256sum "$MODEL" "$SERVER" "$BENCH" >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" perf-start
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }
: >"$OUT/points.jsonl"
: >"$OUT/warmups.jsonl"
nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu.csv" 2>&1 &
sampler_pid=$!

for rep in $(seq 1 5); do
    offset=$(( (rep - 1) * 3 % ${#WIDTH_ORDER[@]} ))
    if (( rep % 2 == 1 )); then
        run_arm "$rep" serial "$offset"
        run_arm "$rep" dual "$(( (offset + 8) % ${#WIDTH_ORDER[@]} ))"
    else
        run_arm "$rep" dual "$offset"
        run_arm "$rep" serial "$(( (offset + 8) % ${#WIDTH_ORDER[@]} ))"
    fi
done

stop_sampler
reduce | tee "$OUT/reduce.log"
snapshot "$OUT/nvidia-smi-after.log" perf-complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
verdict=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hold_floor"]["verdict"])' \
    "$OUT/summary.json")
echo "PERF_PASS verdict=$verdict $(date -u +%FT%TZ)"
trap - EXIT INT TERM
