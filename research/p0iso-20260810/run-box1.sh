#!/usr/bin/env bash
# One invocation is one bounded P0-isolation GPU-lock block on box1.
set -uo pipefail

CONDITION=${1:-}
CELLS=${2:-20}
case "$CONDITION" in
  same|stagger|dedup-off|h2-c2|h2-first-late|h2-c1) ;;
  *) echo "usage: $0 same|stagger|dedup-off|h2-c2|h2-first-late|h2-c1 [cells]" >&2; exit 2 ;;
esac
[[ $CELLS =~ ^[1-9][0-9]*$ ]] || { echo "cells must be positive" >&2; exit 2; }
case "$CONDITION" in
  h2-c1) REQUESTS=1 ;;
  h2-c2) REQUESTS=2 ;;
  *) REQUESTS=8 ;;
esac

REPO=${REPO:-$HOME/memra-cx-grouped}
BIN=${BIN:-$REPO/target/release/memra-server}
WORK_ROOT=${WORK_ROOT:-$HOME/p0iso}
QOS=${QOS:-$WORK_ROOT/harness/qos_probe.py}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-stepfun/step-3.7-flash}
GOLDEN=${GOLDEN:-$HOME/darktrain2/golden-response.bin}
PORT=${PORT:-18431}
BASE=http://127.0.0.1:$PORT
STAMP=${P0ISO_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-$WORK_ROOT/receipts/$STAMP}
OUT=$RUN_ROOT/$CONDITION
EXPECTED_SOURCE=${EXPECTED_SOURCE:-188154299064a42b67fc8eb1f41757cf6237300d}
EXPECTED_BINARY=${EXPECTED_BINARY:-e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3}
EXPECTED_GOLDEN=${EXPECTED_GOLDEN:-21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de}
SERVER_PID=0
SAMPLER_PID=0

mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1

fail() {
  echo "FATAL: $*"
  return 1
}

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
}

snapshot() {
  local path=$1 label=$2
  {
    echo "label=$label"
    echo "ts=$(date -u +%FT%TZ)"
    nvidia-smi \
      --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw \
      --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader
  } >"$path" 2>&1
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
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=0
  fi
}

stop_server() {
  cleanup
  for _ in $(seq 1 90); do
    [[ -z $(compute_apps 2>/dev/null) ]] && return 0
    sleep 1
  done
  compute_apps || true
  fail "GPU processes remained after server shutdown"
}

wait_ready() {
  local log=$1
  for _ in $(seq 1 900); do
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || {
      tail -100 "$log" || true
      return 1
    }
    sleep 1
  done
  tail -100 "$log" || true
  return 1
}

assert_server_clean() {
  local log=$1 failures
  failures=$(grep -Ein \
    "CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|prefix fanout .*FAILED" \
    "$log" || true)
  if [[ -n $failures ]]; then
    echo "$failures"
    fail "server failure signature in $log"
  fi
}

preflight() {
  echo "condition=$CONDITION cells=$CELLS requests=$REQUESTS stamp=$STAMP lock_acquired=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  local source binary golden apps
  source=$(git -C "$REPO" rev-parse HEAD)
  binary=$(sha256sum "$BIN" | awk '{print $1}')
  golden=$(sha256sum "$GOLDEN" | awk '{print $1}')
  echo "source_commit=$source"
  echo "binary_sha256=$binary"
  echo "qos_sha256=$(sha256sum "$QOS" | awk '{print $1}')"
  echo "golden_sha256=$golden"
  git -C "$REPO" status --short --branch
  stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
  snapshot "$OUT/nvidia-smi-before.log" preflight
  [[ $source == "$EXPECTED_SOURCE" ]] || fail "source drift: $source" || return 1
  [[ $binary == "$EXPECTED_BINARY" ]] || fail "binary drift: $binary" || return 1
  [[ $golden == "$EXPECTED_GOLDEN" ]] || fail "golden drift: $golden" || return 1
  apps=$(compute_apps 2>/dev/null || true)
  [[ -z $apps ]] || { echo "$apps"; fail "box1 was not GPU-idle at lock acquisition"; }
}

start_server() {
  local cell=$1 dedup=1
  local log=$cell/server.log
  [[ $CONDITION == dedup-off ]] && dedup=0
  {
    echo "MEMRA_PREFIX_DEDUP=$dedup"
    echo "MEMRA_PREFIX_CACHE_MB=256"
    echo "MEMRA_PRIME_BATCH_HOLD_MS=4"
    echo "MEMRA_TICK_TRACE=1"
    echo "MEMRA_PP_STAGES=2"
    echo "MEMRA_PP_DEVICES=0,1"
    echo "MEMRA_CTX=262144"
    echo "MEMRA_MOE_GROUPED=1"
    echo "MEMRA_PREFILL_TICK=2048"
  } >"$cell/server-env.txt"
  env -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_BG_JOB \
    -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
    MEMRA_COMPAT=openai \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 \
    MEMRA_PREFIX_CACHE_MB=256 \
    MEMRA_PREFIX_DEDUP="$dedup" \
    MEMRA_PRIME_BATCH_HOLD_MS=4 \
    MEMRA_TICK_TRACE=1 \
    "$BIN" >"$log" 2>&1 &
  SERVER_PID=$!
  wait_ready "$log" || fail "server failed readiness for $cell"
}

probe_args() {
  case "$CONDITION" in
    same|dedup-off|h2-c2|h2-c1) ;;
    stagger) printf '%s\n' --stagger-max-ms 200 ;;
    h2-first-late) printf '%s\n' --delays-ms 100,0,0,0,0,0,0,0 ;;
  esac
}

run_cell() {
  local ordinal=$1 cell label probe_rc=0
  cell=$(printf '%s/cell-%02d' "$OUT" "$ordinal")
  label=$(printf '%s-cell-%02d' "$CONDITION" "$ordinal")
  mkdir -p "$cell"
  echo "cell=$label boot_start=$(date -u +%FT%TZ)"
  start_server "$cell" || return 1
  curl -sf "$BASE/metrics" >"$cell/metrics-before.json" || return 1
  snapshot "$cell/serve-ready.log" serve-ready
  nvidia-smi \
    --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$cell/gpu.csv" 2>&1 &
  SAMPLER_PID=$!
  local -a delay_args=()
  while IFS= read -r arg; do
    [[ -n $arg ]] && delay_args+=("$arg")
  done < <(probe_args)
  "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "$label" \
    --requests "$REQUESTS" --max-tokens 64 --golden "$GOLDEN" \
    --rows "$cell/qos-rows.jsonl" --summary "$cell/qos-summary.json" \
    "${delay_args[@]}"
  probe_rc=$?
  echo "$probe_rc" >"$cell/probe-exit-code.txt"
  stop_sampler
  curl -sf "$BASE/metrics" >"$cell/metrics-after.json" || true
  snapshot "$cell/after-probe.log" after-probe
  stop_server || return 1
  assert_server_clean "$cell/server.log" || return 1
  echo "cell=$label probe_rc=$probe_rc boot_done=$(date -u +%FT%TZ)"
  case "$probe_rc" in
    0|86) return "$probe_rc" ;;
    *) return 1 ;;
  esac
}

run_locked() {
  trap cleanup EXIT INT TERM
  preflight || return 1
  local ordinal rc divergent_cells=0
  for ordinal in $(seq 1 "$CELLS"); do
    run_cell "$ordinal"
    rc=$?
    if (( rc == 86 )); then
      divergent_cells=$((divergent_cells + 1))
    elif (( rc != 0 )); then
      echo "block_abort_cell=$ordinal rc=$rc"
      return "$rc"
    fi
  done
  snapshot "$OUT/nvidia-smi-after.log" final
  echo "condition=$CONDITION completed_cells=$CELLS divergent_cells=$divergent_cells block_done=$(date -u +%FT%TZ)"
}

(
  flock -w 60 9 || { echo "LOCK_TIMEOUT"; exit 75; }
  run_locked
) 9>/tmp/memra-gpu.lock
