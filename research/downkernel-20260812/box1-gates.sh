#!/usr/bin/env bash
# One-lock Step-3.7 correctness battery for the downkernel arm.
set -euo pipefail

REPO=${DOWNKERNEL_REPO:-/opt/scratch/nvme/memra-cx-downkernel-base}
BIN=${DOWNKERNEL_BIN:-/opt/scratch/nvme/cx-downkernel-target/release}
OUT=${DOWNKERNEL_OUT:-/opt/scratch/nvme/cx-downkernel-20260812/gates}
MODEL_ROOT=${DOWNKERNEL_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf
PROMPT=$REPO/tools/fast-gate/prompts/probe.txt
KERNEL=$BIN/kernel-check
RUN_GEN=$BIN/run-gen
RUN_SPEC=$BIN/run-spec

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,memory.used,memory.total,temperature.gpu,pstate,clocks.sm,power.draw \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

wait_idle() {
    local apps
    for _ in $(seq 1 120); do
        apps=$(compute_apps || true)
        test -z "$apps" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

run_logged() {
    local label=$1 log=$2
    shift 2
    echo "gate=$label start=$(date -u +%FT%TZ)"
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$log.rc"
    test "$rc" -eq 0
    echo "gate=$label done=$(date -u +%FT%TZ)"
}

for artifact in "$MODEL" "$DRAFT" "$PROMPT" "$KERNEL" "$RUN_GEN" "$RUN_SPEC"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "CORRECTNESS_LOCK_ACQUIRED $(date -u +%FT%TZ)"
cd "$REPO"
echo "host=$(hostname) base_commit=$(git rev-parse HEAD)"
echo "source_diff_sha256=$(git diff | sha256sum | awk '{print $1}')"
git status --short --branch
git diff --check
sha256sum "$MODEL" "$DRAFT" "$KERNEL" "$RUN_GEN" "$RUN_SPEC" >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
test -z "$(compute_apps)" || { compute_apps; echo "FAIL: box1 not GPU-idle"; exit 1; }

run_logged kernel-check "$OUT/kernel-check.log" \
    timeout 3600 env -u MEMRA_KC_FAST CUDA_VISIBLE_DEVICES=0,1 \
    "$KERNEL" "$MODEL"
grep -q 'ALL GREEN' "$OUT/kernel-check.log"
wait_idle

run_logged run-gen "$OUT/run-gen.log" \
    timeout 3600 env CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_NGEN=64 MEMRA_PROMPT_FILE="$PROMPT" "$RUN_GEN" "$MODEL"
grep -q 'prefill argmax=.*decode argmax=.*MATCH' "$OUT/run-gen.log"
grep -q 'batched-prime argmax=.*tokenwise argmax=.*MATCH' "$OUT/run-gen.log"
wait_idle

run_logged run-spec "$OUT/run-spec.log" \
    timeout 3600 env CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
    MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" \
    "$RUN_SPEC" "$MODEL"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"
wait_idle

snapshot "$OUT/nvidia-smi-after.log" complete
test -z "$(compute_apps)" || { compute_apps; echo "FAIL: GPU processes remained"; exit 1; }
echo "CORRECTNESS_PASS $(date -u +%FT%TZ)"
