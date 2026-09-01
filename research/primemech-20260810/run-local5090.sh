#!/usr/bin/env bash
# Local RTX 5090 Laptop concurrent-prime anatomy. No nsys and no mechanism changes.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LANE="$ROOT/research/primemech-20260810"
RAW="$LANE/raw/local5090"
BIN=${BIN:-"$ROOT/target/release/memra-server"}
MODEL=${MODEL:-"$HOME/models/qwen3.5-9b-judge-q8_0.gguf"}
PORT=${PORT:-18127}
BASE="http://127.0.0.1:$PORT"
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/run-$TS.log"
SERVER_PID=
SAMPLE_PID=

mkdir -p "$RAW"

thermal() {
  nvidia-smi \
    --query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu,utilization.memory \
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

boot() {
  local arm=$1
  local tick=$2
  local server_log="$RAW/$arm-server-$TS.log"
  local gpu_log="$RAW/$arm-gpu-$TS.csv"
  local tick_env=()
  if [[ $tick != default ]]; then
    tick_env=("MEMRA_PREFILL_TICK=$tick")
  fi

  env \
    -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_PRIME_PIPE \
    -u MEMRA_PRIME_CHUNK -u MEMRA_MOE_GROUPED -u MEMRA_API_KEY \
    -u MEMRA_PREFILL_TICK \
    "${tick_env[@]}" \
    MEMRA_MODELS="q9=$MODEL" \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_SERVE_BATCH=1 \
    MEMRA_SERVE_SPEC=0 \
    MEMRA_KV_REUSE=0 \
    MEMRA_REUSE_POOL=0 \
    MEMRA_PREFIX_CACHE_MB=0 \
    MEMRA_PREFIX_DEDUP=0 \
    MEMRA_PRIME_BATCH_MAX_T=2048 \
    MEMRA_TICK_TRACE=1 \
    MEMRA_TTFT_TRACE=1 \
    "$BIN" >"$server_log" 2>&1 &
  SERVER_PID=$!

  local ready=0
  for _ in $(seq 1 240); do
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$arm server died during boot; log=$server_log"
      return 1
    fi
    sleep 1
  done
  if [[ $ready != 1 ]]; then
    echo "$arm readiness timeout; log=$server_log"
    return 1
  fi
  echo "$arm ready pid=$SERVER_PID tick=$tick log=$server_log"
  TZ=UTC nvidia-smi dmon -s pucm --gpm-metrics 2,10 -d 1 -o DT \
    --format csv,nounit >"$gpu_log" 2>&1 &
  SAMPLE_PID=$!
  sleep 2
}

run_arm() {
  local arm=$1
  local tick=$2
  local client="$RAW/$arm-client-$TS.jsonl"
  local client_log="$RAW/$arm-client-$TS.log"
  boot "$arm" "$tick" || return 1
  timeout 1200 python3 "$LANE/measure.py" \
    --base "$BASE" \
    --model q9 \
    --label "$arm" \
    --out "$client" \
    --prompt-file "$ROOT/research/step-sku-20260807/prompt-pp4096.txt" \
    --prompt-tokens 4096 \
    --order 1,2,4,4,2,1,2,4,1 \
    --cooldown 2 \
    --timeout 300 >"$client_log" 2>&1
  local rc=$?
  stop_server
  sleep 3
  echo "$arm client rc=$rc"
  thermal
  return "$rc"
}

locked_run() {
  trap stop_server EXIT
  echo "=== entry GPU"
  thermal
  echo "=== entry compute apps"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
  echo "=== production-default scheduler geometry"
  run_arm default default || return 1
  echo "=== explicit 8192-token monolithic outer-tick control"
  run_arm tick8192 8192 || return 1
  python3 "$LANE/analyze.py" \
    --arm default \
      "$RAW/default-client-$TS.jsonl" \
      "$RAW/default-server-$TS.log" \
      "$RAW/default-gpu-$TS.csv" \
    --arm tick8192 \
      "$RAW/tick8192-client-$TS.jsonl" \
      "$RAW/tick8192-server-$TS.log" \
      "$RAW/tick8192-gpu-$TS.csv" \
    --out-prefix "$RAW/anatomy-$TS" >"$RAW/analyze-$TS.log" 2>&1 || return 1
  echo "=== analyzer"
  sed -n '1,80p' "$RAW/analyze-$TS.log"
  echo "=== server fault scan"
  if rg -n -i 'CUDA_ERROR|out of memory|\bOOM\b|panic|NVRM: Xid|CRITICAL:|request error|server died' \
    "$RAW/default-server-$TS.log" "$RAW/tick8192-server-$TS.log"; then
    echo "fault scan found matching lines"
    return 1
  else
    echo "no fault-pattern matches"
  fi
  echo "=== exit GPU"
  thermal
  echo "=== exit compute apps"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
}

{
  echo "=== primemech local5090 ts=$TS commit=$(git rev-parse HEAD)"
  echo "binary=$BIN bytes=$(stat -c %s "$BIN") sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "model=$MODEL bytes=$(stat -c %s "$MODEL") sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
  echo "protocol=q9 Q8_0 chat/no-think, normal-text 4k-class distinct prefixes, generation=8, repeats=3 per c"
  echo "cache controls=KV reuse off, prefix cache off, same-window dedup off"
  echo "arms=production default versus explicit MEMRA_PREFILL_TICK=8192"
  (
    flock -w 60 9 || {
      echo "GPU LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    locked_run
    rc=$?
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  rc=$?
  echo "=== run rc=$rc done=$(date -u +%FT%TZ)"
  exit "$rc"
} >"$SUMMARY" 2>&1
