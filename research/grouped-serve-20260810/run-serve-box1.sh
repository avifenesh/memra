#!/usr/bin/env bash
# Streaming TTFT grouped ON/OFF A/B plus a grouped c=4 cold-prefill burst.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-cx-grouped"}
BIN=${BIN:-"$REPO/target/release/memra-server"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
PROMPT=${PROMPT:-"$REPO/research/step-sku-20260807/prompt-pp4096.txt"}
RAW=${RAW:-"$REPO/research/grouped-serve-20260810/raw/box1/serve"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
N=${N:-5}
BURST_N=${BURST_N:-3}
BURST_ONLY=${BURST_ONLY:-0}
PORT=${PORT:-18127}
BASE="http://127.0.0.1:$PORT"
SUMMARY="$RAW/serve-summary-$TS.log"
CLIENT="$RAW/ttft-client-$TS.jsonl"
BURST="$RAW/burst-c4-$TS.jsonl"
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
    echo "GPU NOT IDLE"
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
  local attempt
  for attempt in $(seq 1 120); do
    if require_idle >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  require_idle
}

boot_server() {
  local label=$1
  local grouped=$2
  local server_log=$3
  env \
    -u MEMRA_PRIME_PP -u MEMRA_PRIME_PIPE -u MEMRA_PRIME_CHUNK \
    -u MEMRA_PRIME_CHUNK_SCHED -u MEMRA_PREFILL_TICK \
    -u MEMRA_SERVE_BATCH -u MEMRA_PRIME_BATCH_HOLD_MS \
    -u MEMRA_MOE_GATE -u MEMRA_MOE_STATS \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MODELS="step35=${MODEL}+${DRAFT}" \
    MEMRA_SERVE_SPEC=0 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 MEMRA_MOE_GROUPED="$grouped" \
    MEMRA_TTFT_TRACE=1 MEMRA_TICK_TRACE=1 \
    MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_TAG="$label" \
    "$BIN" >"$server_log" 2>&1 &
  SERVER_PID=$!

  local attempt
  for attempt in $(seq 1 180); do
    sleep 2
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
      echo "$label ready after <=$((attempt * 2))s log=$server_log"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$label SERVER DIED"
      tail -120 "$server_log"
      return 1
    fi
  done
  echo "$label readiness timeout"
  tail -120 "$server_log"
  return 1
}

run_ttft_arm() {
  local arm=$1
  local pair=$2
  local grouped=0
  [[ "$arm" == grouped ]] && grouped=1
  local server_log="$RAW/server-$arm-p$pair-$TS.log"
  local client_json="$RAW/client-$arm-p$pair-$TS.jsonl"
  local client_log="$RAW/client-$arm-p$pair-$TS.log"

  echo "########## pair=$pair arm=$arm grouped=$grouped ##########"
  snapshot
  boot_server "grouped-serve-$arm-p$pair" "$grouped" "$server_log" || return 1
  python3 research/ttft-20260808/probe.py \
    --base "$BASE" --model step35 --shape 4k --prompt-file "$PROMPT" \
    --requests 1 --warmup 1 --max-tokens 8 --expect-prompt-tokens 4107 \
    --label "grouped-serve-$arm-p$pair" --out "$client_json" \
    --timeout 900 2>&1 | tee "$client_log"
  local probe_rc=${PIPESTATUS[0]}
  stop_server
  wait_idle || return 1
  snapshot
  ((probe_rc == 0)) || return "$probe_rc"

  jq -c --arg arm "$arm" --argjson pair "$pair" \
    '. + {arm: $arm, pair: $pair}' "$client_json" >>"$CLIENT"
}

summarize_ttft() {
  python3 - "$CLIENT" "$N" <<'PY'
import json
import statistics
import sys

path, expected = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path) if line.strip()]
for arm in ("off", "grouped"):
    measured = [row for row in rows if row["arm"] == arm and row["measured"]]
    if len(measured) != expected:
        raise SystemExit(f"{arm}: expected N={expected}, got {len(measured)}")
    prompt_tokens = sorted({row["prompt_tokens"] for row in measured})
    if prompt_tokens != [4107]:
        raise SystemExit(f"{arm}: unexpected prompt token counts {prompt_tokens}")
    values = sorted(row["client_ttft_ms"] / 1000.0 for row in measured)
    print(json.dumps({
        "arm": arm,
        "n": len(values),
        "ttft_p50_s": statistics.median(values),
        "ttft_min_s": values[0],
        "ttft_max_s": values[-1],
        "samples_s": values,
    }, sort_keys=True))
PY
}

run_burst() {
  local server_log="$RAW/server-burst-grouped-$TS.log"
  local client_log="$RAW/burst-c4-$TS.log"
  : >"$BURST"
  echo "########## grouped c=4 burst N=$BURST_N ##########"
  snapshot
  boot_server "grouped-serve-burst-c4" 1 "$server_log" || return 1
  python3 research/grouped-serve-20260810/concurrent_chat_burst.py \
    --base "$BASE" --model step35 --label grouped-serve-burst-c4 \
    --prompt-file "$PROMPT" --expect-prompt-tokens 4107 \
    --out "$BURST" --concurrency 4 --repeats "$BURST_N" \
    --warmup 1 --max-tokens 8 --timeout 900 \
    2>&1 | tee "$client_log"
  local burst_rc=${PIPESTATUS[0]}
  stop_server
  wait_idle || return 1
  snapshot
  ((burst_rc == 0)) || return "$burst_rc"
  if ! grep -Eq '\[step35-batch\].*B=4' "$server_log"; then
    echo "burst assertion FAIL: missing live step35 B=4 batch record"
    return 1
  fi
  echo "burst assertion PASS: live step35 B=4 batch record"
  python3 - "$BURST" "$BURST_N" <<'PY'
import json
import statistics
import sys

path, expected = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path) if line.strip()]
summaries = [row for row in rows
             if row.get("kind") == "summary" and row.get("concurrency") == 4]
if len(summaries) != expected:
    raise SystemExit(f"c4: expected N={expected}, got {len(summaries)}")
walls = sorted(row["wall_to_last_first_token_s"] for row in summaries)
ttft50 = sorted(row["ttft_p50_s"] for row in summaries)
ttft95 = sorted(row["ttft_p95_s"] for row in summaries)
rates = sorted(row["aggregate_prefill_tps"] for row in summaries)
print(json.dumps({
    "arm": "grouped",
    "concurrency": 4,
    "n_bursts": len(summaries),
    "requests": 4 * len(summaries),
    "wall_to_last_first_token_p50_s": statistics.median(walls),
    "burst_ttft_p50_median_s": statistics.median(ttft50),
    "burst_ttft_p95_median_s": statistics.median(ttft95),
    "aggregate_prefill_tps_median": statistics.median(rates),
    "wall_samples_s": walls,
}, sort_keys=True))
PY
}

fault_scan() {
  local found=0 log
  for log in "$RAW"/server-*"$TS".log; do
    [[ -f "$log" ]] || continue
    if grep -Ein 'CUDA[^[:alnum:]]*error|CUDA_ERROR|illegal address|out of memory|panicked at|NVRM: Xid|Xid \(PCI|request error|SERVER DIED' "$log"; then
      found=1
    fi
  done
  if ((found)); then
    echo "server fault scan RED"
    return 1
  fi
  echo "server fault scan GREEN"
}

main() {
  : >"$CLIENT"
  echo "=== grouped-serve serving $TS commit=$(git rev-parse HEAD)"
  echo "binary=$BIN sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "model=$MODEL bytes=$(stat -c %s "$MODEL")"
  echo "draft=$DRAFT bytes=$(stat -c %s "$DRAFT")"
  echo "prompt=$PROMPT bytes=$(stat -c %s "$PROMPT") sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
  echo "config=PP-2 devices 0,1; MEMRA_CTX=262144; spec off; grouped explicit per arm"
  echo "ttft=N=$N/arm, alternating, one warmup per boot, unique cold salts"
  echo "burst=grouped c=4, N=$BURST_N bursts, 4107 cold chat tokens/request"

  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    trap stop_server EXIT
    echo "lock acquired $(date -u +%FT%TZ)"
    snapshot
    require_idle || exit $?

    local pair arm order rc=0
    if ((BURST_ONLY == 0)); then
      for pair in $(seq 1 "$N"); do
        if ((pair % 2 == 1)); then
          order="off grouped"
        else
          order="grouped off"
        fi
        for arm in $order; do
          run_ttft_arm "$arm" "$pair" || rc=1
        done
      done
      summarize_ttft || rc=1
    else
      echo "burst-only recovery: completed TTFT samples are not rerun"
    fi
    run_burst || rc=1
    fault_scan || rc=1
    snapshot
    require_idle || rc=1
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  local serve_rc=$?
  echo "=== grouped-serve serving rc=$serve_rc"
  echo "ttft_client=$CLIENT"
  echo "burst_client=$BURST"
  echo "=== done $(date -u +%FT%TZ)"
  return "$serve_rc"
}

main > >(tee "$SUMMARY") 2>&1
