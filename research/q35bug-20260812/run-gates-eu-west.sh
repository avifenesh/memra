#!/usr/bin/env bash
# Run the Q35 post-fix CLI/decode gates under one exclusive eu-west GPU lock.
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 OUT SERVER RUN_GEN DECODE_BATCH_GATE MODEL PROMPT" >&2
    exit 2
fi

OUT=$1
SERVER=$2
RUN_GEN=$3
DECODE_BATCH_GATE=$4
MODEL=$5
PROMPT=$6

test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1
exec 9>/tmp/memra-gpu.lock
echo "lock_wait_start=$(date -u +%FT%TZ)"
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "lock_acquired=$(date -u +%FT%TZ)"

for artifact in "$SERVER" "$RUN_GEN" "$DECODE_BATCH_GATE" "$MODEL" "$PROMPT"; do
    test -e "$artifact"
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "FAIL: compute applications already active after GPU lock acquisition"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    exit 1
fi

{
    echo "started=$(date -u +%FT%TZ)"
    echo "source_commit=${Q35BUG_SOURCE_COMMIT:-unknown}"
    sha256sum "$SERVER" "$RUN_GEN" "$DECODE_BATCH_GATE" "$MODEL" "$PROMPT"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
} >"$OUT/provenance.txt"

common_env=(
    env
    -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP
    -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE
    -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH
    CUDA_VISIBLE_DEVICES=1
)

set +e
timeout 2400 "${common_env[@]}" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    "$RUN_GEN" "$MODEL" 2>&1 | tee "$OUT/run-gen.log"
run_gen_rc=${PIPESTATUS[0]}
set -e
if [[ "$run_gen_rc" -ne 0 ]]; then
    echo "FAIL: run-gen exited $run_gen_rc"
    exit "$run_gen_rc"
fi
grep -E 'prefill argmax=.*decode argmax=.*MATCH' "$OUT/run-gen.log"
if grep -Eq 'prefill argmax=.*decode argmax=.*MISMATCH' "$OUT/run-gen.log"; then
    echo "FAIL: run-gen argmax mismatch"
    exit 1
fi

set +e
timeout 2400 "${common_env[@]}" "$DECODE_BATCH_GATE" "$MODEL" \
    --steps 32 --batch 2 --mode config 2>&1 | tee "$OUT/decode-batch-config.log"
config_rc=${PIPESTATUS[0]}
set -e
if [[ "$config_rc" -ne 0 ]]; then
    echo "FAIL: decode-batch config gate exited $config_rc"
    exit "$config_rc"
fi
grep -F 'ALL GREEN: decode_step_batch exactness battery' "$OUT/decode-batch-config.log"
grep -F 'gate2 (B=2 vs isolated batched-B=1, bit-checked, 32 steps): PASS' \
    "$OUT/decode-batch-config.log"

set +e
timeout 2400 "${common_env[@]}" MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
    "$DECODE_BATCH_GATE" "$MODEL" --steps 32 --batch 2 --mode strict \
    2>&1 | tee "$OUT/decode-batch-strict.log"
strict_rc=${PIPESTATUS[0]}
set -e
if [[ "$strict_rc" -ne 0 ]]; then
    echo "FAIL: decode-batch strict gate exited $strict_rc"
    exit "$strict_rc"
fi
grep -F 'ALL GREEN: decode_step_batch exactness battery' "$OUT/decode-batch-strict.log"
grep -F 'gate1 (B=1 bit-identity vs decode_step_h, 32 steps, 1 seed(s)): PASS' \
    "$OUT/decode-batch-strict.log"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    >"$OUT/compute-apps-after.txt" || true
echo "completed=$(date -u +%FT%TZ)"
