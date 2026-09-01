#!/usr/bin/env bash
# Concurrent-prefill target-rig battery. No nsys; every command writes a raw log first.
set -uo pipefail

export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
REPO=${REPO:-"$HOME/memra-cx-concprefill"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
RAW=${RAW:-"$REPO/research/concprefill-20260808/raw/box1/gates"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/gates-summary-$TS.log"
PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
PP=("MEMRA_PP_STAGES=2" "MEMRA_PP_DEVICES=0,1")

mkdir -p "$RAW"
cd "$REPO" || exit 1

snapshot() {
  echo "snapshot $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
}

copy_probe_log() {
  local label=$1
  local command_log=$2
  local probe_log
  probe_log=$(grep -oE '/tmp/[[:alnum:]_.-]+\.log' "$command_log" | tail -1)
  if [[ -n "$probe_log" && -f "$probe_log" ]]; then
    local retained="$RAW/$label-probe-$TS.log"
    cp -- "$probe_log" "$retained"
    echo "$label probe_raw=$retained source=$probe_log"
  fi
}

copy_b2_server_log() {
  local label=$1
  local command_log=$2
  local server_log
  server_log=$(sed -n 's/^server log: //p' "$command_log" | tail -1)
  if [[ -n "$server_log" && -f "$server_log" ]]; then
    local retained="$RAW/$label-server-$TS.log"
    cp -- "$server_log" "$retained"
    echo "$label server_raw=$retained source=$server_log"
  fi
}

run_gate() {
  local label=$1
  shift
  local log="$RAW/$label-$TS.log"
  echo "########## $label ##########"
  echo "raw=$log"
  snapshot
  "$@" >"$log" 2>&1
  local gate_rc=$?
  cat "$log"
  copy_probe_log "$label" "$log"
  copy_b2_server_log "$label" "$log"
  echo "$label exit=$gate_rc"
  snapshot
  return "$gate_rc"
}

build_bins() {
  local log="$RAW/build-gate-bins-$TS.log"
  echo "########## build gate binaries ##########"
  set -o pipefail
  cargo build --release -p memra-engine \
    --bin kernel-check \
    --bin concat-prime-probe \
    --bin run-gen \
    --bin run-spec >"$log" 2>&1
  local build_rc=$?
  cat "$log"
  echo "build gate binaries exit=$build_rc"
  return "$build_rc"
}

locked_battery() {
  run_gate kernel-check timeout 3600 \
    ./target/release/kernel-check "$MODEL" || return 1

  run_gate ppsplit env "${PP[@]}" timeout 7200 \
    tools/prime-split-gate.sh "$MODEL" --devices 0,1 --stages 2 \
    --chunks auto,513 --steps 8 || return 1
  run_gate ppsplit-canary env "${PP[@]}" timeout 7200 \
    tools/prime-split-gate.sh "$MODEL" --devices 0,1 --stages 2 \
    --chunks auto,513 --steps 8 --canary || return 1

  run_gate chunkinv35 env "${PP[@]}" timeout 7200 \
    tools/chunk-invariance-gate.sh "$MODEL" --label step35-swa \
    --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
    --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24 || return 1
  run_gate chunkinv35-canary env "${PP[@]}" timeout 7200 \
    tools/chunk-invariance-gate.sh "$MODEL" --label step35-swa \
    --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
    --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24 \
    --canary || return 1

  run_gate tickinv35 env "${PP[@]}" timeout 7200 \
    tools/tick-invariance-gate.sh "$MODEL" --label step35-tick \
    --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
    --budgets 0,1024,513,512,256,64 --splits 64,256,512 \
    --seam MEMRA_PRIME_CALLLOCAL --steps 24 || return 1
  run_gate tickinv35-canary env "${PP[@]}" timeout 7200 \
    tools/tick-invariance-gate.sh "$MODEL" --label step35-tick \
    --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
    --budgets 0,1024,513,512,256,64 --splits 64,256,512 \
    --seam MEMRA_PRIME_CALLLOCAL --steps 24 --canary || return 1

  run_gate run-gen env "${PP[@]}" MEMRA_NGEN=64 timeout 3600 \
    ./target/release/run-gen "$MODEL" --prompt "$PROMPT" || return 1

  run_gate run-spec env "${PP[@]}" MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=32 \
    MEMRA_PROMPT="$PROMPT" timeout 7200 \
    ./target/release/run-spec "$MODEL" || return 1
}

main() {
  echo "=== concurrent-prefill gates $TS commit=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "draft=$DRAFT"
  build_bins || return 1

  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT core battery"
      exit 75
    }
    echo "core lock acquired $(date -u +%FT%TZ)"
    snapshot
    locked_battery || exit 1
    snapshot
    echo "core lock released $(date -u +%FT%TZ)"
  ) 9>/tmp/memra-gpu.lock
  local core_rc=$?
  [[ $core_rc -eq 0 ]] || return "$core_rc"

  # The b2 geometry script owns its own flock window; do not nest it inside the core hold.
  run_gate b2geo35 env MEMRA_STEP37_GGUF="$MODEL" timeout 7200 \
    tools/step35-b2-geometry-gate.sh --port 18119 || return 1
  run_gate b2geo35-canary env MEMRA_STEP37_GGUF="$MODEL" timeout 7200 \
    tools/step35-b2-geometry-gate.sh --canary --port 18120 || return 1

  echo "=== concurrent-prefill gates PASS"
  return 0
}

main >"$SUMMARY" 2>&1
rc=$?
echo "summary=$SUMMARY rc=$rc"
exit "$rc"
