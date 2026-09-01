#!/usr/bin/env bash
# Full local exactness battery for the retained shared-memory swizzle candidate.
# The caller owns /tmp/memra-5090.lock for the entire invocation.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ ${MEMRA_5090_LOCK_HELD:-0} != 1 ]]; then
    echo "FAIL: hold /tmp/memra-5090.lock and set MEMRA_5090_LOCK_HELD=1" >&2
    exit 1
fi

OUT=research/shmconflict-20260813/raw/candidate-v1/gates
CANDIDATE=/home/avifenesh/.cache/memra-targets/cx-shmconflict-candidate-v1/release
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
PROMPT=research/e2e/prompts/pp512.txt
CHUNK_PROMPTS=(
    research/chunk-invariance-20260805/prompt-turn1.txt
    research/chunk-invariance-20260805/prompt-turn2.txt
)

EXPECTED_FLASH=b02e951ebb44aac43220204deac1c88fbd0706131dc965c966a5b1564577ad4b
EXPECTED_KERNEL=562d50557892f0bc52fe59aa3af0f7605dd077eed884fad43920f582e82de43a
EXPECTED_RUN_GEN=1b55812e66e8b991c85a2cbee94452d04292ef880b07b8084dd96a42500f5ce8
EXPECTED_RUN_SPEC=f836700fac1f0e8cea6693a3e3bda42b4bbd5a94025cc4323a23ac24821ebc4c
EXPECTED_CHUNK=13ac1264f5f6f229e9d0f9136b262b4a4ca321039f47d2c3e1f2940849703ae3
EXPECTED_Q27=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
EXPECTED_Q27_DRAFT=b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581
EXPECTED_Q35=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
EXPECTED_Q35_DRAFT=ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a

[[ ! -e $OUT ]] || { echo "FAIL: output already exists: $OUT" >&2; exit 1; }
[[ -z $(git status --porcelain --untracked-files=all) ]] || {
    echo "FAIL: worktree must be clean before the evidence run" >&2
    git status --short >&2
    exit 1
}
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
    nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

assert_idle() {
    local apps
    apps=$(compute_apps)
    if [[ -n $apps ]]; then
        echo "$apps"
        echo "FAIL: GPU 0 is not idle"
        return 1
    fi
}

check_hash() {
    local expected=$1 path=$2 actual
    [[ -r $path ]] || { echo "FAIL: missing $path"; return 1; }
    actual=$(sha256sum "$path" | awk '{print $1}')
    printf '%s  %s\n' "$actual" "$path"
    [[ $actual == "$expected" ]] || {
        echo "FAIL: hash mismatch for $path"
        return 1
    }
}

run_logged() {
    local label=$1 log=$2
    shift 2
    echo "RUN_START label=$label ts=$(date -u +%FT%TZ)"
    set +e
    stdbuf -oL -eL "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "RUN_DONE label=$label rc=$rc ts=$(date -u +%FT%TZ)"
    [[ $rc -eq 0 ]]
    assert_idle
}

assert_kernel() {
    local log=$1 expected=$2
    grep -qx "ALL GREEN ($expected)" "$log"
    ! grep -Eq '(^|[^A-Z])(FAIL|MISMATCH)([^A-Z]|$)' "$log"
}

assert_run_gen() {
    local log=$1
    [[ $(grep -c 'argmax=.*MATCH' "$log") -eq 2 ]]
    grep -q '^prefill argmax=.*decode argmax=.*MATCH$' "$log"
    grep -q '^batched-prime argmax=.*tokenwise argmax=.*MATCH$' "$log"
    ! grep -q 'MISMATCH' "$log"
}

assert_run_spec() {
    local log=$1
    [[ $(grep -c 'self-consistency: PASS' "$log") -eq 8 ]]
    [[ $(grep -cE '^\[generate_spec K=[1-8]\]' "$log") -eq 8 ]]
    grep -q '^=== SELF-CONSISTENCY PASS ===$' "$log"
    ! grep -q 'self-consistency: FAIL\|SELF-CONSISTENCY FAIL\|MISMATCH' "$log"
}

assert_chunkinv() {
    local log=$1
    [[ $(grep -c 'chunkinv verdict: CHUNK-INVARIANT' "$log") -eq 2 ]]
    [[ $(grep -cE '^[[:space:]]+(64|32)[[:space:]]+\| EXACT \|' "$log") -eq 4 ]]
    ! grep -q 'CHUNK-DEPENDENT\|DIFFER\|MISMATCH' "$log"
}

echo "SHMCONFLICT_GATES_START ts=$(date -u +%FT%TZ) pid=$$"
assert_idle
{
    echo "head=$(git rev-parse HEAD)"
    echo "branch=$(git branch --show-current)"
    echo "flash_source_sha256=$EXPECTED_FLASH"
    echo "candidate_target=$CANDIDATE"
    echo "clock_posture=owner-capped 210--1200 MHz; relative evidence only"
    hostname
    uname -a
    nvidia-smi -i 0 --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
} > "$OUT/provenance.txt"

{
    check_hash "$EXPECTED_FLASH" crates/memra-engine/cu/flash_attn.cu
    check_hash "$EXPECTED_KERNEL" "$CANDIDATE/kernel-check"
    check_hash "$EXPECTED_RUN_GEN" "$CANDIDATE/run-gen"
    check_hash "$EXPECTED_RUN_SPEC" "$CANDIDATE/run-spec"
    check_hash "$EXPECTED_CHUNK" "$CANDIDATE/concat-prime-probe"
    check_hash "$EXPECTED_Q27" "$Q27"
    check_hash "$EXPECTED_Q27_DRAFT" "$Q27_DRAFT"
    check_hash "$EXPECTED_Q35" "$Q35"
    check_hash "$EXPECTED_Q35_DRAFT" "$Q35_DRAFT"
    check_hash ce404f9ec20c6aab37220a2428254c6f7dc59286f1620d9060bb30e9d5ad9027 "$PROMPT"
    for prompt in "${CHUNK_PROMPTS[@]}"; do
        sha256sum "$prompt"
    done
} | tee "$OUT/input-sha256.txt"

run_logged kernel-required "$OUT/kernel-required-manifests.log" \
    timeout 2400 "$CANDIDATE/kernel-check" \
        --require-manifest tools/kernel-check-27b.cells \
        --require-manifest tools/kernel-check-step35.cells
assert_kernel "$OUT/kernel-required-manifests.log" '106 cells, 1 skipped'

run_logged kernel-q27 "$OUT/kernel-q27.log" \
    timeout 2400 "$CANDIDATE/kernel-check" "$Q27"
assert_kernel "$OUT/kernel-q27.log" '107 cells, 3 skipped'

run_logged kernel-q35 "$OUT/kernel-q35.log" \
    timeout 2400 "$CANDIDATE/kernel-check" "$Q35"
assert_kernel "$OUT/kernel-q35.log" '113 cells, 1 skipped'

run_logged run-gen-q27 "$OUT/run-gen-q27.log" \
    timeout 2400 env -u MEMRA_FAST -u MEMRA_PROMPT_DIR -u MEMRA_SPEC_K \
        CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
        "$CANDIDATE/run-gen" "$Q27"
assert_run_gen "$OUT/run-gen-q27.log"

run_logged run-gen-q35 "$OUT/run-gen-q35.log" \
    timeout 2400 env -u MEMRA_FAST -u MEMRA_PROMPT_DIR -u MEMRA_SPEC_K \
        CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
        "$CANDIDATE/run-gen" "$Q35"
assert_run_gen "$OUT/run-gen-q35.log"

run_logged run-spec-q27 "$OUT/run-spec-q27.log" \
    timeout 4800 env -u MEMRA_FAST -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_GEN_ONLY \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q27_DRAFT" MEMRA_NGEN=32 \
        MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$CANDIDATE/run-spec" "$Q27"
assert_run_spec "$OUT/run-spec-q27.log"

run_logged run-spec-q35 "$OUT/run-spec-q35.log" \
    timeout 4800 env -u MEMRA_FAST -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_GEN_ONLY \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 \
        MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$CANDIDATE/run-spec" "$Q35"
assert_run_spec "$OUT/run-spec-q35.log"

for model in q27 q35; do
    if [[ $model == q27 ]]; then
        model_path=$Q27
    else
        model_path=$Q35
    fi
    log="$OUT/chunkinv-$model.log"
    echo "RUN_START label=chunkinv-$model ts=$(date -u +%FT%TZ)"
    for prompt in "${CHUNK_PROMPTS[@]}"; do
        env -u MEMRA_FAST -u MEMRA_PRIME_F32CHUNK0 CUDA_VISIBLE_DEVICES=0 \
            "$CANDIDATE/concat-prime-probe" "$model_path" chunkinv \
            --prompt-a "@$prompt" --chunks 2048,64,32 --steps 48 2>&1 | tee -a "$log"
        rc=${PIPESTATUS[0]}
        [[ $rc -eq 0 ]]
    done
    echo "RUN_DONE label=chunkinv-$model rc=0 ts=$(date -u +%FT%TZ)"
    assert_idle
    assert_chunkinv "$log"
done

nvidia-smi -i 0 --query-gpu=index,name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-postflight.txt"
assert_idle
touch "$OUT/PASS"
(
    cd "$OUT"
    find . -type f ! -name MANIFEST.sha256 ! -name driver.log -print0 | sort -z | xargs -0 sha256sum
) > "$OUT/MANIFEST.sha256"
echo "SHMCONFLICT_GATES_PASS ts=$(date -u +%FT%TZ)"
