#!/usr/bin/env bash
# Step35 chunk/tick invariance plus live canary teeth on the clean ring-enabled build.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target-ringval}
PROBE=${PROBE:-$TARGET/release/concat-prime-probe}
MODEL=${MODEL:-$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
PROMPT=${PROMPT:-$REPO/research/chunk-invariance-20260805/prompt-pp6257.txt}
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/invariance-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_PROBE=${EXPECTED_PROBE:-1780a91a898c5aaae1202ac9bea9160c5ce0385756cad599780191717fda5698}

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

run_logged() {
    local label=$1 log=$2 rc
    shift 2
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

copy_tick_raw() {
    local summary=$1 destination=$2 source
    source=$(grep -oE '/tmp/tickinv-gate-[^ )]+\.log' "$summary" | tail -1)
    [[ -n $source && -f $source ]]
    cp "$source" "$destination"
    echo "tick_raw_source=$source destination=$destination"
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    source=$(git -C "$REPO" rev-parse HEAD)
    probe_hash=$(sha256sum "$PROBE" | awk '{print $1}')
    echo "source_commit=$source"
    echo "concat_prime_probe_sha256=$probe_hash"
    echo "target_link=$(readlink -f "$REPO/target")"
    echo "prompt_sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
    git -C "$REPO" status --short --branch --untracked-files=no
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $probe_hash == "$EXPECTED_PROBE" ]]
    [[ $(readlink -f "$REPO/target") == "$TARGET" ]]
    [[ -f $MODEL && -f $PROMPT ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; exit 1; }
    snapshot "$OUT/nvidia-smi-before.log" preflight

    run_logged chunk-naked "$OUT/chunk-naked.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_CHUNKINV_LOG="$OUT/chunk-naked-raw.log" \
        "$REPO/tools/chunk-invariance-gate.sh" "$MODEL" --label step35-swa \
        --prompts "$PROMPT" --chunks 4096,513,512,256,64 \
        --seam MEMRA_STEP35_SWA_TKV --steps 24
    grep -q 'chunk-invariance-gate: PASS' "$OUT/chunk-naked.log"

    run_logged chunk-canary "$OUT/chunk-canary.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        MEMRA_CHUNKINV_LOG="$OUT/chunk-canary-raw.log" \
        "$REPO/tools/chunk-invariance-gate.sh" "$MODEL" --label step35-swa \
        --prompts "$PROMPT" --chunks 4096,513,512,256,64 \
        --seam MEMRA_STEP35_SWA_TKV --steps 24 --canary
    grep -q 'canary broke the assertion as required' "$OUT/chunk-canary.log"

    run_logged tick-naked "$OUT/tick-naked.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        "$REPO/tools/tick-invariance-gate.sh" "$MODEL" --label step35-tick \
        --prompts "$PROMPT" --budgets 0,1024,513,512,256,64 \
        --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24
    grep -q 'tick-invariance-gate: PASS' "$OUT/tick-naked.log"
    copy_tick_raw "$OUT/tick-naked.log" "$OUT/tick-naked-raw.log"

    run_logged tick-canary "$OUT/tick-canary.log" \
        env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        "$REPO/tools/tick-invariance-gate.sh" "$MODEL" --label step35-tick \
        --prompts "$PROMPT" --budgets 0,1024,513,512,256,64 \
        --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24 --canary
    grep -q 'canary broke the assertion as required' "$OUT/tick-canary.log"
    copy_tick_raw "$OUT/tick-canary.log" "$OUT/tick-canary-raw.log"

    snapshot "$OUT/nvidia-smi-after.log" final
    echo 'chunk_naked=PASS'
    echo 'chunk_canary_teeth=PASS'
    echo 'tick_naked=PASS'
    echo 'tick_canary_teeth=PASS'
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
