#!/usr/bin/env bash
# Different-prefix concurrent-prefill anatomy on box1. No nsys; raw server/client/GPU logs.
set -uo pipefail

export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
REPO=${REPO:-"$HOME/memra-cx-concprefill"}
BIN=${BIN:-"$REPO/target/release/memra-server"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
RAW=${RAW:-"$REPO/research/concprefill-20260808/raw/box1"}
PORT=${PORT:-18118}
BASE="http://127.0.0.1:$PORT"
KEY=${KEY:-concprefill-20260808}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/anatomy-$TS.log"
MIXED="$RAW/mixed-client-$TS.jsonl"
CONTROL="$RAW/prime-only-client-$TS.jsonl"
SERVER_PID=
SAMPLE_PID=

mkdir -p "$RAW"
cd "$REPO" || exit 1

thermal() {
  nvidia-smi --query-gpu=index,name,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader
}

stop_sampler() {
  if [[ -n ${SAMPLE_PID:-} ]]; then
    kill "$SAMPLE_PID" 2>/dev/null || true
    wait "$SAMPLE_PID" 2>/dev/null || true
    SAMPLE_PID=
  fi
}

stop_server() {
  stop_sampler
  if [[ -n ${SERVER_PID:-} ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}

sample_gpu() {
  local out=$1
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,utilization.gpu,utilization.memory,memory.used \
      --format=csv,noheader,nounits
    sleep 1
  done >"$out" 2>&1
}

boot() {
  local label=$1
  local server_log="$RAW/$label-server-$TS.log"
  env \
    -u MEMRA_PRIME_PIPE -u MEMRA_PRIME_CHUNK -u MEMRA_PREFILL_TICK \
    -u MEMRA_PRIME_BATCH -u MEMRA_PRIME_BATCH_MAX_T -u MEMRA_PRIME_BATCH_HOLD_MS \
    -u MEMRA_SERVE_BATCH -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_GATE \
    MEMRA_MODELS="step35=${MODEL}+${DRAFT}" \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_API_KEY="$KEY" \
    MEMRA_TTFT_TRACE=1 \
    MEMRA_TICK_TRACE=1 \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    "$BIN" >"$server_log" 2>&1 &
  SERVER_PID=$!
  for attempt in $(seq 1 180); do
    sleep 2
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
      echo "$label ready after <=$((attempt * 2))s log=$server_log"
      sample_gpu "$RAW/$label-gpu-$TS.csv" &
      SAMPLE_PID=$!
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$label server died"
      tail -100 "$server_log"
      return 1
    fi
  done
  echo "$label readiness timeout"
  return 1
}

run_arm() {
  local label=$1
  local background=$2
  local cells=$3
  local out=$4
  boot "$label" || return 1
  python3 research/concprefill-20260808/concurrent_prefill.py \
    --base "$BASE" \
    --model step35 \
    --api-key "$KEY" \
    --label "$label" \
    --out "$out" \
    --cells "$cells" \
    --repeats 3 \
    --prompt-tokens 4096 \
    --background "$background" \
    --timeout 900 || return 1
  stop_server
  sleep 3
  thermal
}

locked_run() {
  trap stop_server EXIT
  echo "=== entry thermal"
  thermal
  echo "=== mixed: c=1/2/4, N=3, decode background c=4"
  run_arm mixed 4 1,2,4 "$MIXED" || return 1
  echo "=== prime-only control: c=4, N=3"
  run_arm prime-only 0 4 "$CONTROL" || return 1
  python3 research/concprefill-20260808/analyze.py \
    --client "$MIXED" \
    --client "$CONTROL" \
    --server "mixed=$RAW/mixed-server-$TS.log" \
    --server "prime-only=$RAW/prime-only-server-$TS.log" \
    --client-table "$RAW/client-table-$TS.tsv" \
    --tick-table "$RAW/tick-table-$TS.tsv"
}

{
  echo "=== concurrent-prefill anatomy ts=$TS commit=$(git rev-parse HEAD)"
  echo "binary=$BIN sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "model=$MODEL bytes=$(stat -c %s "$MODEL")"
  echo "draft=$DRAFT bytes=$(stat -c %s "$DRAFT")"
  echo "trial config: Step trunk+draft, PP-2 dev01, grouped MoE, specplace default"
  (
    flock -w 14400 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    locked_run
    rc=$?
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  rc=$?
  echo "=== anatomy rc=$rc"
  echo "=== done $(date -u +%FT%TZ)"
  exit "$rc"
} >"$SUMMARY" 2>&1
