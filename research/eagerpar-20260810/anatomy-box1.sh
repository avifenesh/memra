#!/usr/bin/env bash
# B=1 Step35 eager-vs-batched anatomy on the designated box1 PP-2 rig.
# One invocation owns one bounded GPU lock window. Raw logs are captured before reduction.
set -euo pipefail

FIXED_REPO=${FIXED_REPO:-$HOME/memra-cx-b1fix}
EAGER_REPO=${EAGER_REPO:-$HOME/memra-cx-grouped}
FIXED_BIN=${FIXED_BIN:-$FIXED_REPO/target/release/decode-batch-bench}
EAGER_BIN=${EAGER_BIN:-$EAGER_REPO/target/release/decode-batch-bench}
MODEL=${MODEL:-$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
STAMP=${EAGERPAR_ANATOMY_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$HOME/eagerpar/anatomy/$STAMP}
EXPECTED_FIXED=${EXPECTED_FIXED:-23cab4ad3b8a82ef86a5e884f6976b35441ff87bf881645b341630db5b568379}

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

run_bench() {
  local label=$1 bin=$2 phase=$3 steps=$4 reps=$5 log
  log=$OUT/$label.log
  echo "cell_start=$label ts=$(date -u +%FT%TZ) phase=$phase steps=$steps reps=$reps"
  snapshot "$OUT/$label-before.log" "$label-before"
  env -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH -u MEMRA_DECODE_BATCH_CAP \
    -u MEMRA_BATCH_PHASE \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 \
    MEMRA_SERVE_SPEC=0 \
    ${phase:+MEMRA_BATCH_PHASE=1} \
    timeout 3600 "$bin" "$MODEL" --steps "$steps" --reps "$reps" --batches 1 --ctx 512 \
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  snapshot "$OUT/$label-after.log" "$label-after"
  echo "cell_done=$label ts=$(date -u +%FT%TZ) rc=$rc"
  return "$rc"
}

(
  flock -w 60 9 || { echo "LOCK_TIMEOUT"; exit 75; }
  echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
  echo "fixed_source=$(git -C "$FIXED_REPO" rev-parse HEAD)"
  echo "eager_source=$(git -C "$EAGER_REPO" rev-parse HEAD)"
  sha256sum "$FIXED_BIN" "$EAGER_BIN" "$MODEL" | tee "$OUT/SHA256SUMS"
  stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL"
  [ "$(sha256sum "$FIXED_BIN" | awk '{print $1}')" = "$EXPECTED_FIXED" ]
  [ -z "$(compute_apps)" ] || { compute_apps; exit 1; }
  snapshot "$OUT/nvidia-smi-before.log" preflight

  # Uninstrumented wall comparison first. Each process includes one discarded warmup rep.
  run_bench wall-batched "$FIXED_BIN" "" 64 3
  run_bench wall-eager "$EAGER_BIN" "" 64 3

  # Diagnostic synchronization deliberately perturbs wall time; only phase rank/share and
  # the q/a-copy bucket are interpreted. Keep the window short and archive the full logs.
  run_bench phase-batched "$FIXED_BIN" 1 8 1
  run_bench phase-eager "$EAGER_BIN" 1 8 1

  snapshot "$OUT/nvidia-smi-after.log" final
  echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
