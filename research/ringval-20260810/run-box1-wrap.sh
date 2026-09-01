#!/usr/bin/env bash
# One bounded box1 block for the Step35 SWA-ring wrap-crossing OFF/ON exactness cell.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target}
RUN_GEN=${RUN_GEN:-$TARGET/release/run-gen}
REPLAY=${REPLAY:-$TARGET/release/replay-acceptance}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
PROMPT=${PROMPT:-$REPO/research/e2e/prompts/p4-16k.txt}
TAPE=${TAPE:-$(cd "$(dirname "$0")" && pwd)/force-tape.txt}
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/wrap-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_RUN_GEN=${EXPECTED_RUN_GEN:-0d5fcfef58230a10e0d92cff2f64d08ad9bd93b7a0d2f2afcc0ccfdca92670b4}
EXPECTED_REPLAY=${EXPECTED_REPLAY:-81e1a3a39a362fa4bdc0d79b18eb12b9d1473f497c30f452ab4ddd9458943430}

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
    local source run_gen_hash replay_hash apps
    source=$(git -C "$REPO" rev-parse HEAD)
    run_gen_hash=$(sha256sum "$RUN_GEN" | awk '{print $1}')
    replay_hash=$(sha256sum "$REPLAY" | awk '{print $1}')
    echo "source_commit=$source"
    echo "run_gen_sha256=$run_gen_hash"
    echo "replay_acceptance_sha256=$replay_hash"
    echo "prompt_sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
    echo "force_tape_sha256=$(sha256sum "$TAPE" | awk '{print $1}')"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT" "$PROMPT" "$TAPE"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $run_gen_hash == "$EXPECTED_RUN_GEN" ]]
    [[ $replay_hash == "$EXPECTED_REPLAY" ]]
    [[ -f $MODEL && -f $DRAFT && -f $PROMPT && -f $TAPE ]]
    [[ $(wc -w <"$TAPE") -ge 16 ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; return 1; }
}

arm_env() {
    local arm=$1
    if [[ $arm == on ]]; then
        printf '%s\n' MEMRA_SWA_RING=1
    fi
}

run_replay_arm() {
    local arm=$1
    local log=$OUT/replay-$arm.log rc
    local -a extra=()
    while IFS= read -r value; do
        [[ -n $value ]] && extra+=("$value")
    done < <(arm_env "$arm")
    echo "replay_arm=$arm start=$(date -u +%FT%TZ)"
    set +e
    timeout 14400 env -u MEMRA_SWA_RING -u MEMRA_CHAT -u MEMRA_SPEC_K \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PRIME_CHUNK=4096 \
        MEMRA_DRAFT="$DRAFT" \
        MEMRA_MTP_DRAFT="$DRAFT" \
        MEMRA_PROMPT_FILE="$PROMPT" \
        MEMRA_REPLAY_T=9216 \
        MEMRA_REPLAY_K=4 \
        MEMRA_REPLAY_STRIDE=256 \
        MEMRA_REPLAY_CHUNK=4096 \
        MEMRA_REPLAY_NLL=1 \
        MEMRA_REPLAY_DUMP="$OUT/replay-$arm.jsonl" \
        "${extra[@]}" \
        "$REPLAY" "$MODEL" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/replay-$arm.rc"
    [[ $rc -eq 0 ]]
    grep -q 'corpus: .* -> 9216 tokens' "$log"
    grep -q 'wrote .* rows' "$log"
    wait_idle
    echo "replay_arm=$arm done=$(date -u +%FT%TZ)"
}

run_gen_arm() {
    local arm=$1
    local log=$OUT/run-gen-$arm.log rc
    local -a extra=()
    while IFS= read -r value; do
        [[ -n $value ]] && extra+=("$value")
    done < <(arm_env "$arm")
    echo "run_gen_arm=$arm start=$(date -u +%FT%TZ)"
    set +e
    timeout 14400 env -u MEMRA_SWA_RING -u MEMRA_CHAT -u MEMRA_DRAFT \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_PRIME_CHUNK=4096 \
        MEMRA_PROMPT_FILE="$PROMPT" \
        MEMRA_NGEN=16 \
        MEMRA_FORCE_TOKENS_FILE="$TAPE" \
        MEMRA_FORCE_LOGITS_AT=15 \
        MEMRA_FORCE_LOGITS_FILE="$OUT/logits-$arm-step15.f32" \
        "${extra[@]}" \
        "$RUN_GEN" "$MODEL" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/run-gen-$arm.rc"
    [[ $rc -eq 0 ]]
    grep -q 'teacher-forced decode: 16 tokens' "$log"
    grep -q 'teacher-forced logits at step 15' "$log"
    grep '^tokens: ' "$log" | tail -1 >"$OUT/tokens-$arm.txt"
    [[ -s $OUT/tokens-$arm.txt && -s $OUT/logits-$arm-step15.f32 ]]
    wait_idle
    echo "run_gen_arm=$arm done=$(date -u +%FT%TZ)"
}

reduce() {
    local replay_cmp=FAIL logits_cmp=FAIL tokens_cmp=FAIL nll_cmp=FAIL
    cmp -s "$OUT/replay-off.jsonl" "$OUT/replay-on.jsonl" && replay_cmp=PASS
    cmp -s "$OUT/logits-off-step15.f32" "$OUT/logits-on-step15.f32" && logits_cmp=PASS
    cmp -s "$OUT/tokens-off.txt" "$OUT/tokens-on.txt" && tokens_cmp=PASS
    grep '^\[replay-nll\]' "$OUT/replay-off.log" >"$OUT/nll-off.txt"
    grep '^\[replay-nll\]' "$OUT/replay-on.log" >"$OUT/nll-on.txt"
    cmp -s "$OUT/nll-off.txt" "$OUT/nll-on.txt" && nll_cmp=PASS
    {
        echo "n=1 per arm"
        echo "prompt_tokens=9216 for replay-acceptance; run-gen uses the complete p4-16k prompt"
        echo "prime_chunk=4096 ring_rows=4639"
        echo "trunk_and_mtp_wrap_guarantee=second 4096-row append crosses the 4639-row physical tail"
        echo "teacher_forced_replay_tokens=$replay_cmp"
        echo "teacher_forced_step15_logits=$logits_cmp"
        echo "forced_output_tokens=$tokens_cmp"
        echo "teacher_forced_nll=$nll_cmp"
        sha256sum "$OUT/replay-off.jsonl" "$OUT/replay-on.jsonl"
        sha256sum "$OUT/logits-off-step15.f32" "$OUT/logits-on-step15.f32"
        wc -c "$OUT/logits-off-step15.f32" "$OUT/logits-on-step15.f32"
        cat "$OUT/nll-off.txt" "$OUT/nll-on.txt"
    } | tee "$OUT/verdict.txt"
    [[ $replay_cmp == PASS && $logits_cmp == PASS && $tokens_cmp == PASS && $nll_cmp == PASS ]]
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    preflight
    snapshot "$OUT/nvidia-smi-before.log" preflight
    run_replay_arm off
    run_replay_arm on
    run_gen_arm off
    run_gen_arm on
    reduce
    snapshot "$OUT/nvidia-smi-after.log" final
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
