#!/usr/bin/env bash
# N=5 interleaved serial-vs-dual c=8/c=16, one box1 lock hold, 512-token decode window.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp0}
OUT=${DUALPP_PERF_OUT:-$REPO/research/dualpp0-20260811/raw/box1/perf}
MODEL_ROOT=${DUALPP_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${DUALPP_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
SERVER=$REPO/target/release/memra-server
BENCH=$REPO/research/newboxgates-20260811/serve_bench.py
PORT=${DUALPP_PERF_PORT:-18459}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37

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
    stop_sampler
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

assert_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|illegal|sentinel' \
        "$log"; then
        return 1
    fi
}

run_point() {
    local rep=$1 arm=$2 width=$3
    local label="r${rep}-${arm}-c${width}"
    echo "point_start=$label ts=$(date -u +%FT%TZ)"
    snapshot "$OUT/$label-thermal-before.log" "$label-before"
    "$BENCH" --base "$BASE" --model "$MODEL_NAME" --shape decode --label "$label" \
        --concurrency "$width" --max-tokens 512 --require-length \
        --out "$OUT/points.jsonl" 2>&1 | tee "$OUT/$label-load.log"
    snapshot "$OUT/$label-thermal-after.log" "$label-after"
    echo "point_done=$label ts=$(date -u +%FT%TZ)"
}

run_arm() {
    local rep=$1 arm=$2 first_width=$3 second_width=$4
    local label="r${rep}-${arm}"
    local log=$OUT/$label-server.log
    local -a policy=(MEMRA_DUAL_PP=0)
    if [[ $arm == dual ]]; then
        policy=(MEMRA_DUAL_PP=1 MEMRA_PP_OVERLAP=1)
    fi
    echo "arm_start=$label ts=$(date -u +%FT%TZ) widths=$first_width,$second_width"
    {
        echo "arm=$arm"
        echo "MEMRA_DUAL_PP=$([[ $arm == dual ]] && echo 1 || echo 0)"
        echo "MEMRA_PP_OVERLAP=$([[ $arm == dual ]] && echo 1 || echo '<unset>')"
        echo "MEMRA_SERVE_SPEC=0"
        echo "MEMRA_PP_STAGES=2"
        echo "MEMRA_PP_DEVICES=0,1"
        echo "MEMRA_MOE_GROUPED=1"
        echo "MEMRA_PREFIX_CACHE_MB=0"
    } >"$OUT/$label-env.txt"
    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_DUAL_PP_TIMING \
        -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_SIG_ROUTER \
        -u MEMRA_MOE_DEV -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_BG_JOB \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_SWA_RING "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_TAG="dualpp-$label" "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
    "$BENCH" --base "$BASE" --model "$MODEL_NAME" --shape warmup \
        --label "$label-warmup" --concurrency 1 --max-tokens 16 \
        --out "$OUT/warmups.jsonl" >"$OUT/$label-warmup.log" 2>&1
    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/$label-gpu.csv" 2>&1 &
    sampler_pid=$!
    run_point "$rep" "$arm" "$first_width"
    run_point "$rep" "$arm" "$second_width"
    curl -sf "$BASE/metrics" >"$OUT/$label-metrics.txt"
    stop_server
    assert_clean "$log"
    if [[ $arm == dual ]]; then
        grep -q '\[dual-pp\] dual-active PP-2 decode engaged' "$log"
    else
        ! grep -q '\[dual-pp\]' "$log"
    fi
    echo "arm_done=$label ts=$(date -u +%FT%TZ)"
}

# CUDA events perturb host issue slightly, so the frozen N=5 score above is instrument-free.
# This companion process runs under the same lock and records cumulative event spans before and
# after one full c8 and c16 window. Deltas provide per-wave/stage wall times without contaminating
# the kill-rule denominator.
run_timing_diagnostics() {
    local label=timing-dual
    local log=$OUT/$label-server.log
    echo "timing_start=$label ts=$(date -u +%FT%TZ)"
    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_DUAL_PP_TIMING \
        -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_SIG_ROUTER -u MEMRA_MOE_DEV -u MEMRA_SERVE_BATCH \
        -u MEMRA_SPEC_K -u MEMRA_BG_JOB -u MEMRA_DECODE_BATCH_CAP -u MEMRA_SWA_RING \
        MEMRA_DUAL_PP=1 MEMRA_PP_OVERLAP=1 MEMRA_DUAL_PP_TIMING=1 \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_TAG="dualpp-$label" "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
    "$BENCH" --base "$BASE" --model "$MODEL_NAME" --shape warmup \
        --label "$label-warmup" --concurrency 1 --max-tokens 16 \
        --out "$OUT/timing-warmup.jsonl" >"$OUT/$label-warmup.log" 2>&1
    : >"$OUT/timing-points.jsonl"
    for width in 8 16; do
        curl -sf "$BASE/metrics" >"$OUT/timing-c$width-before.json"
        snapshot "$OUT/timing-c$width-thermal-before.log" "timing-c$width-before"
        "$BENCH" --base "$BASE" --model "$MODEL_NAME" --shape decode \
            --label "timing-dual-c$width" --concurrency "$width" --max-tokens 512 \
            --require-length --out "$OUT/timing-points.jsonl" \
            2>&1 | tee "$OUT/timing-c$width-load.log"
        snapshot "$OUT/timing-c$width-thermal-after.log" "timing-c$width-after"
        curl -sf "$BASE/metrics" >"$OUT/timing-c$width-after.json"
    done
    stop_server
    assert_clean "$log"
    grep -q '\[dual-pp\] dual-active PP-2 decode engaged' "$log"
    echo "timing_done=$label ts=$(date -u +%FT%TZ)"
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
rows = [json.loads(line) for line in points_path.read_text().splitlines()]
requests = [row for row in rows if row.get("kind") == "request"]
points = [row for row in rows if row.get("kind") == "summary"]
assert len(points) == 20, len(points)
assert len(requests) == 240, len(requests)
assert all(row["ok"] for row in requests), [row for row in requests if not row["ok"]]
assert all(point["n_error"] == 0 for point in points), points
assert all(point["completion_tokens_total"] == point["concurrency"] * 512 for point in points)
assert all(point["finish_reasons"] == ["length"] for point in points)

def series(arm, width):
    suffix = f"-{arm}-c{width}"
    return [float(row["decode_window_tok_s"]) for row in points if row["label"].endswith(suffix)]

widths = {}
for width in (8, 16):
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

temperatures = []
clocks = []
for path in out.glob("*-gpu.csv"):
    with path.open(newline="", errors="replace") as stream:
        for row in csv.reader(stream):
            if len(row) < 7:
                continue
            try:
                temperatures.append(float(row[3].strip()))
                clocks.append(float(row[6].strip()))
            except ValueError:
                pass

c16_delta = widths["c16"]["delta_pct"]

stage_names = ("wave_a_stage0", "wave_a_stage1", "wave_b_stage0", "wave_b_stage1")

def timing_snapshot(path):
    row = json.loads(path.read_text())
    dual = row.get("dual_pp", {})
    spans = dual.get("cuda_event_spans", {})
    return {
        "overlaps": int(dual.get("overlaps", 0)),
        "spans": {
            name: {
                "samples": int(spans.get(name, {}).get("samples", 0)),
                "total_ms": float(spans.get(name, {}).get("total_ms", 0.0)),
            }
            for name in stage_names
        },
    }

stage_diagnostics = {}
for width in (8, 16):
    before = timing_snapshot(out / f"timing-c{width}-before.json")
    after = timing_snapshot(out / f"timing-c{width}-after.json")
    spans = {}
    for name in stage_names:
        samples = after["spans"][name]["samples"] - before["spans"][name]["samples"]
        total_ms = after["spans"][name]["total_ms"] - before["spans"][name]["total_ms"]
        assert samples > 0, (width, name, before, after)
        spans[name] = {
            "samples": samples,
            "total_ms": total_ms,
            "mean_ms": total_ms / samples,
        }
    assert len({span["samples"] for span in spans.values()}) == 1, (width, spans)
    overlaps = after["overlaps"] - before["overlaps"]
    assert overlaps > 0, (width, before, after)
    stage0_ms = statistics.mean((spans["wave_a_stage0"]["mean_ms"],
                                 spans["wave_b_stage0"]["mean_ms"]))
    stage1_ms = statistics.mean((spans["wave_a_stage1"]["mean_ms"],
                                 spans["wave_b_stage1"]["mean_ms"]))
    stage_diagnostics[f"c{width}"] = {
        "method": "unscored companion run; CUDA events bracket layer ranges",
        "overlap_counter_delta": overlaps,
        "spans": spans,
        "stage0_mean_ms": stage0_ms,
        "stage1_mean_ms": stage1_ms,
        "stage_balance_ratio": max(stage0_ms, stage1_ms) / min(stage0_ms, stage1_ms),
    }

c16_diag = stage_diagnostics["c16"]
if c16_delta >= 15.0:
    kill_diagnosis = "gain-clears-hold-floor"
elif c16_diag["overlap_counter_delta"] == 0:
    kill_diagnosis = "dead-overlap-mechanics"
elif c16_diag["stage_balance_ratio"] >= 1.25:
    kill_diagnosis = "stage-load-imbalance"
else:
    kill_diagnosis = "genuine-no-win-at-balanced-stage-load"

summary = {
    "schema": "memra.dualpp0.perf.v1",
    "source_commit": sys.argv[3],
    "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition",
    "protocol": "N=5 interleaved arms, rotating width order, one lock hold; unscored CUDA-event diagnostic companion",
    "metric": "aggregate completion tokens after first visible token / decode window second",
    "max_tokens_per_request": 512,
    "widths": widths,
    "dual_stage_diagnostics": stage_diagnostics,
    "thermal_regime": {
        "artificial_cooldown": False,
        "sample_interval_ms": 250,
        "samples": len(temperatures),
        "temperature_c_min": min(temperatures) if temperatures else None,
        "temperature_c_max": max(temperatures) if temperatures else None,
        "sm_clock_mhz_min": min(clocks) if clocks else None,
        "sm_clock_mhz_max": max(clocks) if clocks else None,
    },
    "kill_rule": {
        "threshold_pct_at_c16": 15.0,
        "observed_pct": c16_delta,
        "verdict": "KILL" if c16_delta < 15.0 else "HOLD",
        "diagnosis": kill_diagnosis,
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

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "PERF_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
git status --short --branch --untracked-files=no
sha256sum "$MODEL" "$SERVER" "$BENCH" >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }
: >"$OUT/points.jsonl"
: >"$OUT/warmups.jsonl"

for rep in $(seq 1 5); do
    if (( rep % 2 == 1 )); then
        run_arm "$rep" serial 8 16
        run_arm "$rep" dual 16 8
    else
        run_arm "$rep" dual 8 16
        run_arm "$rep" serial 16 8
    fi
done

run_timing_diagnostics
reduce | tee "$OUT/reduce.log"
snapshot "$OUT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
verdict=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["kill_rule"]["verdict"])' \
    "$OUT/summary.json")
echo "PERF_PASS verdict=$verdict $(date -u +%FT%TZ)"
