#!/usr/bin/env bash
# Fixed-binary target-rig core battery with MEMRA_SWA_RING=1.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target-ringval}
KERNEL=${KERNEL:-$TARGET/release/kernel-check}
DECODE_BATCH=${DECODE_BATCH:-$TARGET/release/decode-batch-gate}
RUN_GEN=${RUN_GEN:-$TARGET/release/run-gen}
RUN_SPEC=${RUN_SPEC:-$TARGET/release/run-spec}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PROMPT=${PROMPT:-$REPO/tools/fast-gate/prompts/probe.txt}
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/core-gates-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_KERNEL=${EXPECTED_KERNEL:-c499498820478f2f355955ae0b963855dd1378142c4d83251f6980c1ab54588a}
EXPECTED_DECODE_BATCH=${EXPECTED_DECODE_BATCH:-52b18f8453c683de69c4bd0605597316bae9b1d5378034332aaabdae02a89f3c}
EXPECTED_RUN_GEN=${EXPECTED_RUN_GEN:-833b11f7e6f76e014bd94bf17f3543f3354d6ab23624b7dfc98bfb3d63eeefd5}
EXPECTED_RUN_SPEC=${EXPECTED_RUN_SPEC:-942d0d1260ad44e8e349b1049c49d47faf692ffede318452972cef8a751100d1}

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
    } >"$path" 2>&1
}

wait_idle() {
    local _
    for _ in $(seq 1 180); do
        [[ -z $(compute_apps) ]] && return 0
        sleep 1
    done
    compute_apps
    return 1
}

preflight() {
    local source kernel_hash decode_batch_hash run_gen_hash run_spec_hash apps
    source=$(git -C "$REPO" rev-parse HEAD)
    kernel_hash=$(sha256sum "$KERNEL" | awk '{print $1}')
    decode_batch_hash=$(sha256sum "$DECODE_BATCH" | awk '{print $1}')
    run_gen_hash=$(sha256sum "$RUN_GEN" | awk '{print $1}')
    run_spec_hash=$(sha256sum "$RUN_SPEC" | awk '{print $1}')
    echo "source_commit=$source"
    echo "kernel_check_sha256=$kernel_hash"
    echo "decode_batch_gate_sha256=$decode_batch_hash"
    echo "run_gen_sha256=$run_gen_hash"
    echo "run_spec_sha256=$run_spec_hash"
    echo "prompt_sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT" "$PROMPT"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $kernel_hash == "$EXPECTED_KERNEL" ]]
    [[ $decode_batch_hash == "$EXPECTED_DECODE_BATCH" ]]
    [[ $run_gen_hash == "$EXPECTED_RUN_GEN" ]]
    [[ $run_spec_hash == "$EXPECTED_RUN_SPEC" ]]
    [[ -x $DECODE_BATCH ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; return 1; }
}

run_logged() {
    local label=$1 log=$2
    shift 2
    local rc
    echo "gate=$label start=$(date -u +%FT%TZ)"
    set +e
    timeout 14400 "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/$label.rc"
    wait_idle
    [[ $rc -eq 0 ]]
    echo "gate=$label done=$(date -u +%FT%TZ)"
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    preflight
    snapshot "$OUT/nvidia-smi-before.log" preflight

    run_logged kernel-check "$OUT/kernel-check.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 "$KERNEL"
    grep -q '^ALL GREEN ([0-9][0-9]* cells, [0-9][0-9]* skipped)$' "$OUT/kernel-check.log"

    run_logged decode-batch-gate "$OUT/decode-batch-gate.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
        "$DECODE_BATCH" "$MODEL" --mode pp --batch 1,2,4,8 --steps 24 --reps 2 \
        --stages 2 --plen 520
    grep -q 'pp mode verdict: 0 failing arm(s)' "$OUT/decode-batch-gate.log"
    grep -q 'ALL GREEN: batched PP-2 stage-split exactness battery' "$OUT/decode-batch-gate.log"

    run_logged run-gen "$OUT/run-gen.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_NGEN=64 MEMRA_PROMPT_FILE="$PROMPT" \
        "$RUN_GEN" "$MODEL"
    grep -q 'prefill argmax=.*decode argmax=.*MATCH' "$OUT/run-gen.log"
    grep -q 'batched-prime argmax=.*tokenwise argmax=.*MATCH' "$OUT/run-gen.log"

    run_logged run-spec "$OUT/run-spec.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" \
        "$RUN_SPEC" "$MODEL"
    [[ $(grep -c 'self-consistency: PASS' "$OUT/run-spec.log") -eq 8 ]]
    grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec.log"

    snapshot "$OUT/nvidia-smi-after.log" final
    echo 'core_gate_verdict=PASS'
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
