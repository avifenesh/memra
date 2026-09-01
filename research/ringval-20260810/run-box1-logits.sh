#!/usr/bin/env bash
# Fixed-binary OFF/ON teacher-forced logit comparison after the Step35 SWA ring wraps.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target}
RUN_GEN=${RUN_GEN:-$TARGET/release/run-gen}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPT_IDS=${PROMPT_IDS:-$HERE/prompt-9216.ids}
TAPE=${TAPE:-$HERE/force-tape.txt}
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/logits-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_RUN_GEN=${EXPECTED_RUN_GEN:-0d5fcfef58230a10e0d92cff2f64d08ad9bd93b7a0d2f2afcc0ccfdca92670b4}
EXPECTED_PROMPT=${EXPECTED_PROMPT:-68a4ef669e393fb0fd7a45bcefae4f097d31583e1db4b3edf73bf7d658fa7b3a}

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
    local source run_gen_hash prompt_hash apps
    source=$(git -C "$REPO" rev-parse HEAD)
    run_gen_hash=$(sha256sum "$RUN_GEN" | awk '{print $1}')
    prompt_hash=$(sha256sum "$PROMPT_IDS" | awk '{print $1}')
    echo "source_commit=$source"
    echo "run_gen_sha256=$run_gen_hash"
    echo "prompt_ids_sha256=$prompt_hash"
    echo "force_tape_sha256=$(sha256sum "$TAPE" | awk '{print $1}')"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$PROMPT_IDS" "$TAPE"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $run_gen_hash == "$EXPECTED_RUN_GEN" ]]
    [[ $prompt_hash == "$EXPECTED_PROMPT" ]]
    [[ $(wc -w <"$PROMPT_IDS") -eq 9216 ]]
    [[ $(wc -w <"$TAPE") -ge 16 ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; return 1; }
}

run_arm() {
    local arm=$1 log=$OUT/run-gen-$1.log rc
    local -a extra=()
    local -a prompt_ids=()
    read -r -a prompt_ids <"$PROMPT_IDS"
    [[ ${#prompt_ids[@]} -eq 9216 ]]
    if [[ $arm == on ]]; then
        extra+=(MEMRA_SWA_RING=1)
    fi
    echo "arm=$arm start=$(date -u +%FT%TZ) prompt_tokens=${#prompt_ids[@]}"
    set +e
    timeout 14400 env -u MEMRA_SWA_RING -u MEMRA_CHAT -u MEMRA_PROMPT_FILE \
        -u MEMRA_PROMPT -u MEMRA_DRAFT -u MEMRA_MTP_DRAFT \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PRIME_CHUNK=4096 \
        MEMRA_PRIME_GATE=0 \
        MEMRA_NGEN=16 \
        MEMRA_FORCE_TOKENS_FILE="$TAPE" \
        MEMRA_FORCE_LOGITS_AT=15 \
        MEMRA_FORCE_LOGITS_FILE="$OUT/logits-$arm-step15.f32" \
        "${extra[@]}" \
        "$RUN_GEN" "$MODEL" "${prompt_ids[@]}" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/run-gen-$arm.rc"
    wait_idle
    [[ $rc -eq 0 ]]
    grep -q 'teacher-forced logits at step 15' "$log"
    grep '^tokens: ' "$log" | tail -1 >"$OUT/tokens-$arm.txt"
    grep '^teacher-forced EXACTNESS:' "$log" >"$OUT/exactness-$arm.txt"
    [[ -s $OUT/logits-$arm-step15.f32 ]]
    echo "arm=$arm done=$(date -u +%FT%TZ)"
}

reduce() {
    local logits_cmp=FAIL tokens_cmp=FAIL exactness_cmp=FAIL
    cmp -s "$OUT/logits-off-step15.f32" "$OUT/logits-on-step15.f32" && logits_cmp=PASS
    cmp -s "$OUT/tokens-off.txt" "$OUT/tokens-on.txt" && tokens_cmp=PASS
    cmp -s "$OUT/exactness-off.txt" "$OUT/exactness-on.txt" && exactness_cmp=PASS
    {
        echo "n=1 per arm"
        echo "prompt_tokens=9216"
        echo "prime_chunk=4096 ring_rows=4639"
        echo "trunk_wrap_guarantee=sequential teacher-forced cache crosses physical row 4639"
        echo "teacher_forced_step15_full_logits=$logits_cmp"
        echo "forced_output_tokens=$tokens_cmp"
        echo "teacher_forced_summary=$exactness_cmp"
        sha256sum "$OUT/logits-off-step15.f32" "$OUT/logits-on-step15.f32"
        wc -c "$OUT/logits-off-step15.f32" "$OUT/logits-on-step15.f32"
        cat "$OUT/exactness-off.txt" "$OUT/exactness-on.txt"
    } | tee "$OUT/verdict.txt"
    [[ $logits_cmp == PASS && $tokens_cmp == PASS && $exactness_cmp == PASS ]]
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    preflight
    snapshot "$OUT/nvidia-smi-before.log" preflight
    run_arm off
    run_arm on
    reduce
    snapshot "$OUT/nvidia-smi-after.log" final
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
