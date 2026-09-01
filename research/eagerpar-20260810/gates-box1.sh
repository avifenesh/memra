#!/usr/bin/env bash
# Candidate correctness batteries on box1. Each invocation owns one bounded GPU lock.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
  core|generation) ;;
  *) echo "usage: $0 core|generation" >&2; exit 2 ;;
esac

REPO=${REPO:-$HOME/memra-cx-eagerpar}
MODEL=${MODEL:-$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf}
SOURCE=${SOURCE:-711fbcaaef54491d22488a84d40b7fc35e5a58dd}
STAMP=${EAGERPAR_GATES_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$HOME/eagerpar/gates/$STAMP/$MODE}
PROMPT='Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard.'

KERNEL_BIN=$REPO/target/release/kernel-check
BATCH_BIN=$REPO/target/release/decode-batch-gate
GEN_BIN=$REPO/target/release/run-gen
SPEC_BIN=$REPO/target/release/run-spec
KERNEL_SHA=a65b650c15de01a08f4acb4c9a6c095846de8c2e55c6d98c84dbcbe1364f4359
BATCH_SHA=f40f9611e98ac3c4b14a46a40234a27fdffc827e17978463ec2c1a7ea2c37814
GEN_SHA=9d4dd6a53c713d736784351e105b8acc620ff77b3a04e5523e6069a2a79e8a4b
SPEC_SHA=6690f91817bd993a7abb1a069b8b277adf6cb277a951c6fd2ff13d543f4f065f

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

check_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  echo "binary=$path sha256=$actual"
  [ "$actual" = "$expected" ]
}

run_logged() {
  local label=$1
  shift
  echo "gate_start=$label ts=$(date -u +%FT%TZ)"
  set +e
  "$@" 2>&1 | tee "$OUT/$label.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "gate_done=$label ts=$(date -u +%FT%TZ) rc=$rc"
  return "$rc"
}

(
  flock -w 60 9 || { echo "LOCK_TIMEOUT"; exit 75; }
  echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) mode=$MODE stamp=$STAMP"
  echo "source=$(git -C "$REPO" rev-parse HEAD)"
  [ "$(git -C "$REPO" rev-parse HEAD)" = "$SOURCE" ]
  [ -f "$MODEL" ] && [ -f "$DRAFT" ]
  stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
  [ -z "$(compute_apps)" ] || { compute_apps; exit 1; }
  snapshot "$OUT/nvidia-smi-before.log" preflight

  if [ "$MODE" = core ]; then
    check_hash "$KERNEL_BIN" "$KERNEL_SHA"
    check_hash "$BATCH_BIN" "$BATCH_SHA"
    run_logged kernel-check timeout 3600 "$KERNEL_BIN" "$MODEL"
    grep -q '^ALL GREEN ([0-9][0-9]* cells, [0-9][0-9]* skipped)$' "$OUT/kernel-check.log"
    run_logged decode-batch-gate env \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
      MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
      timeout 7200 "$BATCH_BIN" "$MODEL" --mode pp --batch 1,2,4,8 \
      --steps 24 --reps 2 --stages 2 --plen 520
    grep -q 'ALL GREEN: batched PP-2 stage-split exactness battery' \
      "$OUT/decode-batch-gate.log"
  else
    check_hash "$GEN_BIN" "$GEN_SHA"
    check_hash "$SPEC_BIN" "$SPEC_SHA"
    run_logged run-gen env \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
      MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_NGEN=64 \
      timeout 3600 "$GEN_BIN" "$MODEL" --prompt "$PROMPT"
    [ "$(grep -c 'MATCH' "$OUT/run-gen.log")" -ge 2 ]
    run_logged run-spec env \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
      MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_NGEN=32 \
      MEMRA_MTP_DRAFT="$DRAFT" MEMRA_PROMPT="$PROMPT" \
      timeout 7200 "$SPEC_BIN" "$MODEL"
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"
    [ "$(grep -c 'self-consistency: PASS' "$OUT/run-spec.log")" -eq 8 ]
  fi

  snapshot "$OUT/nvidia-smi-after.log" final
  echo "lock_released=$(date -u +%FT%TZ) result=PASS"
) 9>/tmp/memra-gpu.lock
