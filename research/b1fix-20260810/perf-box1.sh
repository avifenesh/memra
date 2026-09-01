#!/usr/bin/env bash
# Interleaved box1 A/B for the PP-N Step35 B=1 correctness default.
# One invocation owns one GPU lock window and runs five reps per binary in alternating order.
set -euo pipefail

REPO=${REPO:-$HOME/memra-cx-b1fix}
FIXED_BIN=${FIXED_BIN:-$REPO/target/release/memra-server}
BASE_REPO=${BASE_REPO:-$HOME/memra-cx-grouped}
BASE_BIN=${BASE_BIN:-$BASE_REPO/target/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-stepfun/step-3.7-flash}
PORT=${PORT:-18432}
BASE=http://127.0.0.1:$PORT
STAMP=${B1FIX_PERF_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$HOME/b1fix/perf/$STAMP}
EXPECTED_FIXED=${EXPECTED_FIXED:-6a7c2046eb3197773def91baf012abd629e0b0ced239ec2d38016c93be5ca7e5}
EXPECTED_BASE=${EXPECTED_BASE:-e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3}
EXPECTED_BASE_SOURCE=${EXPECTED_BASE_SOURCE:-188154299064a42b67fc8eb1f41757cf6237300d}
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
  local fixed_hash base_hash apps
  fixed_hash=$(sha256sum "$FIXED_BIN" | awk '{print $1}')
  base_hash=$(sha256sum "$BASE_BIN" | awk '{print $1}')
  echo "fixed_source=$(git -C "$REPO" rev-parse HEAD)"
  echo "fixed_binary_sha256=$fixed_hash"
  echo "base_source=$(git -C "$BASE_REPO" rev-parse HEAD)"
  echo "base_binary_sha256=$base_hash"
  echo "load_serve_sha256=$(sha256sum "$REPO/tools/load-serve.py" | awk '{print $1}')"
  git -C "$REPO" merge-base --is-ancestor 2689e5bf HEAD
  [ "$fixed_hash" = "$EXPECTED_FIXED" ]
  [ "$base_hash" = "$EXPECTED_BASE" ]
  [ "$(git -C "$BASE_REPO" rev-parse HEAD)" = "$EXPECTED_BASE_SOURCE" ]
  [ -f "$MODEL" ] && [ -f "$DRAFT" ]
  stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
  apps=$(compute_apps)
  [ -z "$apps" ] || { echo "$apps"; return 1; }
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

  python3 "$REPO/tools/load-serve.py" \
    --base "$BASE" --model "$MODEL_NAME" --concurrency 1 --requests 1 \
    --max-tokens 8 --greedy --stream --warmup 1 --timeout 1800 \
    --label "$label-ttft" --out "$OUT/points.jsonl" \
    --per-request "$OUT/requests.jsonl"
  python3 "$REPO/tools/load-serve.py" \
    --base "$BASE" --model "$MODEL_NAME" --concurrency 1 --requests 1 \
    --max-tokens 256 --greedy --stream --warmup 0 --timeout 1800 \
    --label "$label-decode" --out "$OUT/points.jsonl" \
    --per-request "$OUT/requests.jsonl"

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
      run_arm base "$rep" "$BASE_BIN"
      run_arm fixed "$rep" "$FIXED_BIN"
    else
      run_arm fixed "$rep" "$FIXED_BIN"
      run_arm base "$rep" "$BASE_BIN"
    fi
  done

  snapshot "$OUT/nvidia-smi-after.log" final
  echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
