#!/usr/bin/env bash
# Interleaved current-batched vs B=1-specialized live-server A/B on box1.
# One invocation owns one GPU lock and runs five reps per arm in alternating order.
set -euo pipefail

CANDIDATE_REPO=${CANDIDATE_REPO:-$HOME/memra-cx-eagerpar}
CURRENT_REPO=${CURRENT_REPO:-$HOME/memra-cx-b1fix}
CANDIDATE_BIN=${CANDIDATE_BIN:-$CANDIDATE_REPO/target/release/memra-server}
CURRENT_BIN=${CURRENT_BIN:-$CURRENT_REPO/target/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-stepfun/step-3.7-flash}
PORT=${PORT:-18435}
BASE=http://127.0.0.1:$PORT
STAMP=${EAGERPAR_PERF_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$HOME/eagerpar/perf/$STAMP}
EXPECTED_CANDIDATE=${EXPECTED_CANDIDATE:-43ad098d46bb26d644ba0b742d92f3f014d9287ac72e8a0edb8ebf9dac3ba608}
EXPECTED_CURRENT=${EXPECTED_CURRENT:-6a7c2046eb3197773def91baf012abd629e0b0ced239ec2d38016c93be5ca7e5}
EXPECTED_CANDIDATE_SOURCE=${EXPECTED_CANDIDATE_SOURCE:-711fbcaaef54491d22488a84d40b7fc35e5a58dd}
EXPECTED_CURRENT_SOURCE=${EXPECTED_CURRENT_SOURCE:-2ef6d75cbf9ff7a09f685bcc1fc54b84bb8f81fb}
SERVER_PID=0

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
      --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw,power.limit \
      --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader
  } > "$path" 2>&1
}

cleanup() {
  if (( SERVER_PID > 0 )); then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=0
  fi
}

wait_idle() {
  local _
  for _ in $(seq 1 90); do
    [ -z "$(compute_apps)" ] && return 0
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
      tail -100 "$log"
      return 1
    fi
    sleep 1
  done
  tail -100 "$log"
  return 1
}

assert_server_clean() {
  local log=$1
  if grep -Ein \
    'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|prefix fanout .*FAILED' \
    "$log"; then
    echo "FATAL: server failure signature in $log"
    return 1
  fi
}

preflight() {
  local candidate_hash current_hash apps
  candidate_hash=$(sha256sum "$CANDIDATE_BIN" | awk '{print $1}')
  current_hash=$(sha256sum "$CURRENT_BIN" | awk '{print $1}')
  echo "candidate_source=$(git -C "$CANDIDATE_REPO" rev-parse HEAD)"
  echo "candidate_binary_sha256=$candidate_hash"
  echo "current_source=$(git -C "$CURRENT_REPO" rev-parse HEAD)"
  echo "current_binary_sha256=$current_hash"
  echo "load_serve_sha256=$(sha256sum "$CANDIDATE_REPO/tools/load-serve.py" | awk '{print $1}')"
  [ "$(git -C "$CANDIDATE_REPO" rev-parse HEAD)" = "$EXPECTED_CANDIDATE_SOURCE" ]
  [ "$(git -C "$CURRENT_REPO" rev-parse HEAD)" = "$EXPECTED_CURRENT_SOURCE" ]
  [ "$candidate_hash" = "$EXPECTED_CANDIDATE" ]
  [ "$current_hash" = "$EXPECTED_CURRENT" ]
  [ -f "$MODEL" ] && [ -f "$DRAFT" ]
  stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
  apps=$(compute_apps)
  [ -z "$apps" ] || { echo "$apps"; return 1; }
}

load_point() {
  local label=$1 concurrency=$2 requests=$3 max_tokens=$4 warmup=$5
  python3 "$CANDIDATE_REPO/tools/load-serve.py" \
    --base "$BASE" --model "$MODEL_NAME" --concurrency "$concurrency" \
    --requests "$requests" --max-tokens "$max_tokens" --greedy --stream \
    --warmup "$warmup" --timeout 1800 --label "$label" \
    --out "$OUT/points.jsonl" --per-request "$OUT/requests.jsonl"
}

run_arm() {
  local arm=$1 rep=$2 bin=$3 label log
  label=$(printf '%s-r%02d' "$arm" "$rep")
  log=$OUT/server-$label.log
  echo "arm_start=$label ts=$(date -u +%FT%TZ)"
  snapshot "$OUT/thermal-$label-before.log" "$label-before"

  env -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH -u MEMRA_SERVE_BATCH \
    -u MEMRA_DECODE_BATCH_CAP -u MEMRA_SPEC_K -u MEMRA_BG_JOB \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
    MEMRA_COMPAT=openai \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_SERVE_SPEC=0 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 \
    "$bin" > "$log" 2>&1 &
  SERVER_PID=$!
  wait_ready "$log"
  grep -m1 'decode chunk cap' "$log" || true

  load_point "$label-ttft" 1 1 8 1
  load_point "$label-c1" 1 1 256 0
  load_point "$label-c2" 2 2 256 0
  load_point "$label-c4" 4 4 256 0

  cleanup
  wait_idle
  assert_server_clean "$log"
  snapshot "$OUT/thermal-$label-after.log" "$label-after"
  echo "arm_done=$label ts=$(date -u +%FT%TZ)"
}

(
  flock -w 60 9 || { echo "LOCK_TIMEOUT"; exit 75; }
  trap cleanup EXIT INT TERM
  echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
  preflight
  snapshot "$OUT/nvidia-smi-before.log" preflight

  for rep in $(seq 1 5); do
    if (( rep % 2 == 1 )); then
      run_arm current "$rep" "$CURRENT_BIN"
      run_arm candidate "$rep" "$CANDIDATE_BIN"
    else
      run_arm candidate "$rep" "$CANDIDATE_BIN"
      run_arm current "$rep" "$CURRENT_BIN"
    fi
  done

  python3 - "$OUT/points.jsonl" "$OUT/requests.jsonl" <<'PY'
import json
import sys

points = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
requests = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]
assert len(points) == 40, len(points)
assert all(point["n_err"] == 0 and point["n_shed"] == 0 for point in points)
assert all(request["ok"] for request in requests)
for point in points:
    if point["label"].endswith("-ttft"):
        assert point["completion_tokens_total"] == 8
    else:
        assert point["completion_tokens_total"] == point["concurrency"] * 256
print(f"receipt_check=PASS points={len(points)} request_rows={len(requests)}")
PY

  snapshot "$OUT/nvidia-smi-after.log" final
  echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
