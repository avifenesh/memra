#!/usr/bin/env bash
# Lever-B serving A/B: unsplit reference vs naked PP-2 pipeline, alternating
# arms under one GPU-lock hold. Every command writes raw output before summary
# parsing; one warmup precedes each measured request after each server boot.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-pp2pipe"}
BIN=${BIN:-"$REPO/target/release/memra-server"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
PROMPT=${PROMPT:-"$REPO/research/step-sku-20260807/prompt-pp4096.txt"}
RAW=${RAW:-"$REPO/research/pp2pipe-20260809/raw/box1/ttft"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
N=${N:-5}
PORT=${PORT:-18125}
BASE="http://127.0.0.1:$PORT"
SUMMARY="$RAW/ttft-ab-summary-$TS.log"
CLIENT="$RAW/ttft-ab-client-$TS.jsonl"
SERVER_PID=

mkdir -p "$RAW"
cd "$REPO" || exit 1

snapshot() {
  echo "snapshot $(date -u +%FT%TZ)"
  nvidia-smi \
    --query-gpu=index,name,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
}

require_idle() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader 2>/dev/null || true)
  if [[ -n "$apps" ]]; then
    echo "GPU NOT IDLE AFTER LOCK ACQUISITION"
    printf '%s\n' "$apps"
    return 76
  fi
}

stop_server() {
  if [[ -n ${SERVER_PID:-} ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}

wait_idle() {
  local attempt apps
  for attempt in $(seq 1 120); do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader 2>/dev/null || true)
    if [[ -z "$apps" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "GPU applications remained after server stop"
  printf '%s\n' "$apps"
  return 1
}

boot_server() {
  local arm=$1
  local pair=$2
  local server_log=$3
  local arm_env=()
  if [[ "$arm" == "unsplit" ]]; then
    arm_env+=("MEMRA_PRIME_PP=0")
  fi

  env \
    -u MEMRA_PRIME_PP -u MEMRA_PRIME_PIPE -u MEMRA_PRIME_CHUNK \
    -u MEMRA_PRIME_CHUNK_SCHED -u MEMRA_PREFILL_TICK \
    -u MEMRA_MOE_GROUPED -u MEMRA_SERVE_BATCH \
    -u MEMRA_PRIME_BATCH_HOLD_MS \
    "${arm_env[@]}" \
    MEMRA_MODELS="step35=${MODEL}+${DRAFT}" \
    MEMRA_SERVE_SPEC=0 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_TTFT_TRACE=1 \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_TAG="pp2pipe-${arm}-p${pair}" \
    "$BIN" >"$server_log" 2>&1 &
  SERVER_PID=$!

  local attempt
  for attempt in $(seq 1 180); do
    sleep 2
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
      echo "$arm pair=$pair ready after <=$((attempt * 2))s log=$server_log"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$arm pair=$pair SERVER DIED"
      tail -100 "$server_log"
      return 1
    fi
  done
  echo "$arm pair=$pair readiness timeout"
  tail -100 "$server_log"
  return 1
}

run_arm() {
  local arm=$1
  local pair=$2
  local server_log="$RAW/server-${arm}-p${pair}-$TS.log"
  local client_log="$RAW/client-${arm}-p${pair}-$TS.jsonl"
  local console_log="$RAW/client-${arm}-p${pair}-$TS.log"

  echo "########## pair=$pair arm=$arm ##########"
  snapshot
  boot_server "$arm" "$pair" "$server_log" || return 1
  python3 research/ttft-20260808/probe.py \
    --base "$BASE" \
    --model step35 \
    --shape 4k \
    --prompt-file "$PROMPT" \
    --requests 1 \
    --warmup 1 \
    --max-tokens 8 \
    --expect-prompt-tokens 4107 \
    --label "pp2pipe-${arm}-p${pair}" \
    --out "$client_log" \
    --timeout 900 2>&1 | tee "$console_log"
  local probe_rc=${PIPESTATUS[0]}
  stop_server
  wait_idle || return 1
  snapshot
  (( probe_rc == 0 )) || return "$probe_rc"

  jq -c --arg arm "$arm" --argjson pair "$pair" \
    '. + {arm: $arm, pair: $pair}' "$client_log" >>"$CLIENT"
}

summarize() {
  python3 - "$CLIENT" "$N" <<'PY'
import json
import statistics
import sys

path, expected = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path) if line.strip()]
for arm in ("unsplit", "pipe"):
    measured = [row for row in rows if row["arm"] == arm and row["measured"]]
    if len(measured) != expected:
        raise SystemExit(f"{arm}: expected N={expected}, got {len(measured)}")
    prompt_tokens = sorted({row["prompt_tokens"] for row in measured})
    if prompt_tokens != [4107]:
        raise SystemExit(f"{arm}: unexpected prompt token counts {prompt_tokens}")
    values = sorted(row["client_ttft_ms"] / 1000.0 for row in measured)
    summary = {
        "arm": arm,
        "n": len(values),
        "ttft_p50_s": statistics.median(values),
        "ttft_min_s": values[0],
        "ttft_max_s": values[-1],
        "samples_s": values,
    }
    print(json.dumps(summary, sort_keys=True))
    if arm == "pipe" and summary["ttft_p50_s"] >= 10.0:
        raise SystemExit("PIPE TTFT target failed: p50 must be below 10 seconds")
PY
}

main() {
  : >"$CLIENT"
  echo "=== pp2pipe TTFT A/B $TS commit=$(git rev-parse HEAD)"
  echo "binary=$BIN sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "model=$MODEL bytes=$(stat -c %s "$MODEL") sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
  echo "draft=$DRAFT bytes=$(stat -c %s "$DRAFT") sha256=$(sha256sum "$DRAFT" | awk '{print $1}')"
  echo "prompt=$PROMPT bytes=$(stat -c %s "$PROMPT") sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
  echo "protocol=alternating unsplit/default PIPE, one warmup per boot, N=$N measured per arm"
  echo "serve=PP-2 devices 0,1; spec off; naked grouped/microchunk/solo-prefill defaults"

  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    trap stop_server EXIT
    echo "lock acquired $(date -u +%FT%TZ)"
    snapshot
    require_idle || exit $?

    local pair arm order
    for pair in $(seq 1 "$N"); do
      if (( pair % 2 == 1 )); then
        order="unsplit pipe"
      else
        order="pipe unsplit"
      fi
      for arm in $order; do
        run_arm "$arm" "$pair" || exit 1
      done
    done

    summarize || exit 1
    snapshot
    echo "lock released $(date -u +%FT%TZ)"
  ) 9>/tmp/memra-gpu.lock
  local trial_rc=$?
  echo "=== pp2pipe TTFT A/B rc=$trial_rc"
  echo "client_jsonl=$CLIENT"
  echo "=== done $(date -u +%FT%TZ)"
  return "$trial_rc"
}

exec > >(tee "$SUMMARY") 2>&1
main
