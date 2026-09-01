#!/usr/bin/env bash
# Interleaved eagerpar-baseline vs packed-row candidate live-server A/B on box1.
set -euo pipefail

: "${EXPECTED_CANDIDATE_SOURCE:?set the staged rowwalk commit}"
: "${EXPECTED_CANDIDATE:?set the candidate memra-server SHA-256}"
: "${EXPECTED_CURRENT_SOURCE:?set the eagerpar baseline commit}"
: "${EXPECTED_CURRENT:?set the eagerpar baseline memra-server SHA-256}"
: "${GOLDEN_RECEIPT:?set the prior candidate golden-receipt.json path}"

CANDIDATE_REPO=${CANDIDATE_REPO:-$HOME/memra-cx-rowwalk}
CURRENT_REPO=${CURRENT_REPO:-$HOME/memra-cx-eagerpar}
CANDIDATE_BIN=${CANDIDATE_BIN:-$CANDIDATE_REPO/target/release/memra-server}
CURRENT_BIN=${CURRENT_BIN:-$CURRENT_REPO/target/release/memra-server}
LOAD_SERVE=${LOAD_SERVE:-$CANDIDATE_REPO/tools/load-serve.py}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-stepfun/step-3.7-flash}
PORT=${PORT:-18437}
BASE=http://127.0.0.1:$PORT
STAMP=${ROWWALK_PERF_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$HOME/rowwalk/perf/$STAMP}
SERVER_PID=0
SAMPLER_PID=0

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
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
  } > "$path" 2>&1
}

stop_sampler() {
  if (( SAMPLER_PID > 0 )); then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=0
  fi
}

cleanup() {
  stop_sampler
  if (( SERVER_PID > 0 )); then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=0
  fi
}

wait_idle() {
  local _
  for _ in $(seq 1 120); do
    [[ -z $(compute_apps) ]] && return 0
    sleep 1
  done
  compute_apps
  return 1
}

wait_ready() {
  local log=$1 _
  for _ in $(seq 1 900); do
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -120 "$log"
      return 1
    fi
    sleep 1
  done
  tail -120 "$log"
  return 1
}

assert_server_clean() {
  local log=$1
  if grep -Ein \
    'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|prefix fanout .*FAILED' \
    "$log"; then
    return 1
  fi
  grep -q '\[step35-batch\] first B>1 batched step35 walk' "$log"
}

check_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  echo "binary=$path sha256=$actual"
  [[ $actual == "$expected" ]]
}

preflight() {
  local apps
  echo "candidate_source=$(git -C "$CANDIDATE_REPO" rev-parse HEAD)"
  echo "current_source=$(git -C "$CURRENT_REPO" rev-parse HEAD)"
  [[ $(git -C "$CANDIDATE_REPO" rev-parse HEAD) == "$EXPECTED_CANDIDATE_SOURCE" ]]
  [[ $(git -C "$CURRENT_REPO" rev-parse HEAD) == "$EXPECTED_CURRENT_SOURCE" ]]
  check_hash "$CANDIDATE_BIN" "$EXPECTED_CANDIDATE"
  check_hash "$CURRENT_BIN" "$EXPECTED_CURRENT"
  echo "load_serve_sha256=$(sha256sum "$LOAD_SERVE" | awk '{print $1}')"
  stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT" "$GOLDEN_RECEIPT"
  python3 - "$GOLDEN_RECEIPT" "$EXPECTED_CANDIDATE_SOURCE" "$EXPECTED_CANDIDATE" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
assert row["candidate_source"] == sys.argv[2], row
assert row["candidate_binary_sha256"] == sys.argv[3], row
summary = row["summary"]
assert summary["exactness"] == "match", row
assert summary["golden_matches"] == 1 and summary["golden_divergences"] == 0, row
print("prior_candidate_golden=PASS")
PY
  apps=$(compute_apps)
  [[ -z $apps ]] || { echo "$apps"; return 1; }
}

load_point() {
  local label=$1 concurrency=$2 requests=$3 max_tokens=$4 warmup=$5
  python3 "$LOAD_SERVE" --base "$BASE" --model "$MODEL_NAME" \
    --concurrency "$concurrency" --requests "$requests" --max-tokens "$max_tokens" \
    --greedy --stream --warmup "$warmup" --timeout 1800 --label "$label" \
    --out "$OUT/points.jsonl" --per-request "$OUT/requests.jsonl"
}

run_arm() {
  local arm=$1 rep=$2 bin=$3 label log
  label=$(printf '%s-r%02d' "$arm" "$rep")
  log=$OUT/server-$label.log
  echo "arm_start=$label ts=$(date -u +%FT%TZ) target_shape=plain-decode gamma=1"
  snapshot "$OUT/thermal-$label-before.log" "$label-before"
  env -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH -u MEMRA_SERVE_BATCH \
    -u MEMRA_DECODE_BATCH_CAP -u MEMRA_SPEC_K -u MEMRA_BG_JOB \
    CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
    MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=256 \
    "$bin" > "$log" 2>&1 &
  SERVER_PID=$!
  wait_ready "$log"

  nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 500 > "$OUT/gpu-$label.csv" 2>&1 &
  SAMPLER_PID=$!
  load_point "$label-warmup" 1 1 8 1
  load_point "$label-c2" 2 2 256 0
  load_point "$label-c4" 4 4 256 0
  load_point "$label-c8" 8 8 256 0
  stop_sampler
  cleanup
  wait_idle
  assert_server_clean "$log"
  snapshot "$OUT/thermal-$label-after.log" "$label-after"
  echo "arm_done=$label ts=$(date -u +%FT%TZ)"
}

reduce_receipt() {
  python3 - "$OUT/points.jsonl" "$OUT/requests.jsonl" "$OUT" \
    "$EXPECTED_CANDIDATE_SOURCE" "$EXPECTED_CANDIDATE" \
    "$EXPECTED_CURRENT_SOURCE" "$EXPECTED_CURRENT" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

points_path, requests_path, out_path = map(pathlib.Path, sys.argv[1:4])
points = [json.loads(line) for line in points_path.read_text().splitlines()]
requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
assert len(points) == 40, len(points)
assert all(p["n_err"] == 0 and p["n_shed"] == 0 for p in points), points
assert all(r["ok"] for r in requests), requests
for p in points:
    if p["label"].endswith("-warmup"):
        assert p["completion_tokens_total"] == 8, p
    else:
        assert p["completion_tokens_total"] == p["concurrency"] * 256, p

def series(arm, width):
    suffix = f"-c{width}"
    return [p["agg_tok_s"] for p in points if p["label"].startswith(arm + "-") and p["label"].endswith(suffix)]

metrics = {}
for width in (2, 4, 8):
    current = series("current", width)
    candidate = series("candidate", width)
    assert len(current) == len(candidate) == 5, (width, current, candidate)
    cm = statistics.median(current)
    nm = statistics.median(candidate)
    metrics[f"c{width}"] = {
        "N": 5,
        "current_aggregate_tok_s": current,
        "candidate_aggregate_tok_s": candidate,
        "current_median_tok_s": cm,
        "candidate_median_tok_s": nm,
        "delta_pct": (nm / cm - 1.0) * 100.0,
    }

temperatures = []
for path in out_path.glob("gpu-*.csv"):
    with path.open(newline="", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4:
                try:
                    temperatures.append(float(row[3].strip()))
                except ValueError:
                    pass

summary = {
    "schema": "memra.rowwalk.perf-summary.v1",
    "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition",
    "protocol": "N=5 paired rounds, alternating arm order, fresh server per arm",
    "metric": "aggregate completion tokens / live wall second",
    "provenance": {
        "candidate_source": sys.argv[4],
        "candidate_binary_sha256": sys.argv[5],
        "current_source": sys.argv[6],
        "current_binary_sha256": sys.argv[7],
    },
    "widths": metrics,
    "thermal_regime": {
        "sampling_interval_ms": 500,
        "sample_count": len(temperatures),
        "temperature_c_min": min(temperatures) if temperatures else None,
        "temperature_c_max": max(temperatures) if temperatures else None,
        "artificial_cooldown": False,
    },
    "receipt": {
        "points": len(points),
        "request_rows": len(requests),
        "errors": 0,
        "short_completions": 0,
    },
    "moesd_target_timing": {
        "status": "not-measured-by-live-rowwalk-harness",
        "reason": "T_T(B,gamma) requires the standalone B-by-gamma verify harness and expert-union telemetry; ordinary serve wall is not target-forward time",
        "rows": [
            {"B": width, "T_T_B_1_ms": None, "gamma": None, "T_T_B_gamma_ms": None}
            for width in (2, 4, 8)
        ],
    },
}
(out_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"receipt_check=PASS points={len(points)} request_rows={len(requests)}")
PY
}

(
  flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
  trap cleanup EXIT INT TERM
  echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
  preflight
  snapshot "$OUT/nvidia-smi-before.log" preflight
  : > "$OUT/points.jsonl"
  : > "$OUT/requests.jsonl"
  for rep in $(seq 1 5); do
    if (( rep % 2 == 1 )); then
      run_arm current "$rep" "$CURRENT_BIN"
      run_arm candidate "$rep" "$CANDIDATE_BIN"
    else
      run_arm candidate "$rep" "$CANDIDATE_BIN"
      run_arm current "$rep" "$CURRENT_BIN"
    fi
  done
  reduce_receipt
  snapshot "$OUT/nvidia-smi-after.log" final
  echo "lock_released=$(date -u +%FT%TZ) result=PASS"
) 9>/tmp/memra-gpu.lock
