#!/usr/bin/env bash
# cx-coldfix local 5090 correctness battery. The caller owns no GPU process.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${1:-research/coldfix-20260812/raw/battery}
MODELS=/data/ai-ml/hf-models
Q27=$MODELS/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_DRAFT=$MODELS/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q35=$MODELS/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=$MODELS/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
PROMPT=research/e2e/prompts/pp512.txt

test ! -e "$OUT" || { echo "battery output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
for path in "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" "$PROMPT"; do
    test -f "$path" || { echo "missing battery input: $path" >&2; exit 2; }
done

run_logged() {
    local label=$1 log=$2
    shift 2
    echo "RUN_START label=$label ts=$(date -u +%FT%TZ)"
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "RUN_DONE label=$label rc=$rc ts=$(date -u +%FT%TZ)"
    return "$rc"
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.current.sm,clocks.max.sm,power.draw,power.limit,memory.used,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    } 2>&1 | tee "$path"
}

exec 9>/tmp/memra-gpu.lock
flock -w 55 9 || { echo "memra GPU lock busy" >&2; exit 75; }
exec 8>/tmp/gpu5090.lock
flock -w 55 8 || { echo "5090 GPU lock busy" >&2; exit 75; }

snapshot "$OUT/gpu-before.log" before
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" \
    || { echo "GPU not idle after lock acquisition" >&2; exit 76; }
{
    echo "source=$(git rev-parse HEAD)"
    echo "worker_tree=$(git hash-object crates/memra-server/src/worker.rs)"
    echo "serve_smoke_tree=$(git hash-object tools/serve-smoke.sh)"
    git status --short
    rustc --version
    cargo --version
    nvcc --version
    sha256sum "$Q27" "$Q27_DRAFT" "$Q35" "$Q35_DRAFT" "$PROMPT" \
        research/sellgate-20260812/workload.lock.json tools/q35-cold-mixed-gate.py
} 2>&1 | tee "$OUT/provenance.log"

run_logged build "$OUT/build.log" cargo build --release
run_logged cargo-test "$OUT/cargo-test.log" cargo test

run_logged kernel-check "$OUT/kernel-check.log" \
    target/release/kernel-check \
    --require-manifest tools/kernel-check-27b.cells \
    --require-manifest tools/kernel-check-step35.cells
grep -q '^ALL GREEN' "$OUT/kernel-check.log"
! grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$OUT/kernel-check.log"

run_logged run-gen-q27 "$OUT/run-gen-q27.log" env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    target/release/run-gen "$Q27"
grep -q 'argmax=.*MATCH' "$OUT/run-gen-q27.log"
! grep -q 'MISMATCH' "$OUT/run-gen-q27.log"

run_logged run-gen-q35 "$OUT/run-gen-q35.log" env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
    target/release/run-gen "$Q35"
grep -q 'argmax=.*MATCH' "$OUT/run-gen-q35.log"
! grep -q 'MISMATCH' "$OUT/run-gen-q35.log"

run_logged run-spec-q27 "$OUT/run-spec-q27.log" env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
    CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q27_DRAFT" MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 target/release/run-spec "$Q27"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-q27.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-q27.log"
! grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-q27.log"

run_logged run-spec-q35 "$OUT/run-spec-q35.log" env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
    CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 \
    MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 target/release/run-spec "$Q35"
test "$(grep -c 'self-consistency: PASS' "$OUT/run-spec-q35.log")" -eq 8
grep -q '=== SELF-CONSISTENCY PASS ===' "$OUT/run-spec-q35.log"
! grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec-q35.log"

run_logged serve-smoke "$OUT/serve-smoke.log" env \
    MEMRA_Q35_COLD_MODEL="$Q35" bash tools/serve-smoke.sh
grep -q 'Q35 mixed c=4: 20/20 requests reached exactly 60 tokens' "$OUT/serve-smoke.log"
grep -q 'Q35 routed-MoE carried prime batches remain gated' "$OUT/serve-smoke.log"
grep -q 'serve-smoke: 0 failed' "$OUT/serve-smoke.log"
test -f /tmp/serve-smoke.log && cp /tmp/serve-smoke.log "$OUT/serve-smoke-q35-server.log"
test -f /tmp/serve-smoke-q35-cold-mixed.log \
    && cp /tmp/serve-smoke-q35-cold-mixed.log "$OUT/serve-smoke-q35-cell.log"

run_logged serve-stress "$OUT/serve-stress.log" env \
    MEMRA_STRESS_LOG="$OUT/serve-stress-server.log" \
    MEMRA_STRESS_ROWS="$OUT/serve-stress-rows.jsonl" \
    bash tools/serve-stress-gate.sh
grep -q 'serve-stress-gate: ALL GREEN' "$OUT/serve-stress.log"

snapshot "$OUT/gpu-after.log" after
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" \
    || { echo "GPU process remained after battery" >&2; exit 77; }
touch "$OUT/battery.ok"
(
    cd "$OUT"
    find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"$OUT/MANIFEST.sha256"
echo "COLD_FIX_BATTERY_ALL_GREEN ts=$(date -u +%FT%TZ)"
