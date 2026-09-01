#!/usr/bin/env bash
# Explicit-grouped correctness gates for the PP-2 serving promotion decision.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-cx-grouped"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
RAW=${RAW:-"$REPO/research/grouped-serve-20260810/raw/box1/gates"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/gates-summary-$TS.log"
PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
PP=("CUDA_VISIBLE_DEVICES=0,1" "MEMRA_PP_STAGES=2" "MEMRA_PP_DEVICES=0,1")

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

run_gate() {
  local label=$1
  shift
  local log="$RAW/$label-$TS.log"
  echo "########## $label ##########"
  echo "raw=$log"
  snapshot
  "$@" >"$log" 2>&1
  local gate_rc=$?
  if ((gate_rc == 0)); then
    tail -40 "$log"
  else
    tail -120 "$log"
  fi
  echo "$label exit=$gate_rc"
  snapshot
  return "$gate_rc"
}

require_log() {
  local label=$1
  local pattern=$2
  local log="$RAW/$label-$TS.log"
  if grep -Eq "$pattern" "$log"; then
    echo "$label assertion PASS: $pattern"
    return 0
  fi
  echo "$label assertion FAIL: missing $pattern in $log"
  return 1
}

main() {
  echo "=== grouped-serve gates $TS commit=$(git rev-parse HEAD)"
  echo "run-gen=$(sha256sum target/release/run-gen | awk '{print $1}')"
  echo "kernel-check=$(sha256sum target/release/kernel-check | awk '{print $1}')"
  echo "model=$MODEL bytes=$(stat -c %s "$MODEL")"
  echo "config=MEMRA_MOE_GROUPED=1 PP-2 devices 0,1"
  local rc=0

  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    snapshot
    require_idle || exit $?

    run_gate grouped-oracle env "${PP[@]}" \
      MEMRA_MOE_GROUPED=1 MEMRA_MOE_GATE=1 MEMRA_MOE_STATS=1 \
      MEMRA_NGEN=1 timeout 3600 \
      ./target/release/run-gen "$MODEL" --prompt "$PROMPT" || rc=1

    local identical mismatch
    identical=$(grep -c 'moe-gate .* BYTE-IDENTICAL' \
      "$RAW/grouped-oracle-$TS.log" || true)
    mismatch=$(grep -c 'MISMATCH' "$RAW/grouped-oracle-$TS.log" || true)
    echo "grouped-oracle byte_identical=$identical mismatch=$mismatch"
    [[ "$identical" == 210 ]] || {
      echo "grouped-oracle assertion FAIL: expected 210 BYTE-IDENTICAL rows"
      rc=1
    }
    [[ "$mismatch" == 0 ]] || rc=1
    require_log grouped-oracle 'dispatch=resident-q8-rows' || rc=1
    require_log grouped-oracle 'dispatch=resident-q8-clamped-pairs' || rc=1

    run_gate kernel-check env "${PP[@]}" MEMRA_MOE_GROUPED=1 \
      timeout 3600 ./target/release/kernel-check "$MODEL" || rc=1
    require_log kernel-check '^ALL GREEN ([0-9]+ cells, [0-9]+ skipped)$' || rc=1

    run_gate run-gen env "${PP[@]}" MEMRA_MOE_GROUPED=1 MEMRA_NGEN=64 \
      timeout 3600 ./target/release/run-gen "$MODEL" --prompt "$PROMPT" || rc=1
    require_log run-gen 'prefill argmax=.* MATCH$' || rc=1
    require_log run-gen 'batched-prime argmax=.* MATCH$' || rc=1

    snapshot
    require_idle || rc=1
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  local gate_rc=$?
  echo "=== grouped-serve gates rc=$gate_rc"
  echo "=== done $(date -u +%FT%TZ)"
  return "$gate_rc"
}

main > >(tee "$SUMMARY") 2>&1
