#!/usr/bin/env bash
# Full post-change correctness receipt. Call under the shared 5090 lock or let this script
# acquire it non-blocking; stdout/stderr are tee'd before any verdict parsing.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$REPO/research/budgetsize-20260813/raw/gates
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf

cd "$REPO"
if [ "${MEMRA_5090_LOCK_HELD:-0}" != 1 ]; then
    exec 9>/tmp/memra-5090.lock
    flock -n 9 || { echo "FAIL: /tmp/memra-5090.lock is held" >&2; exit 75; }
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | grep -q '[0-9]'; then
    echo "FAIL: a compute process exists after acquiring the 5090 lock"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    exit 1
fi
test -f "$Q27"
test -f "$Q35"
mkdir -p "$OUT"

set +e
env -u MEMRA_PREFIX_CACHE_MB TMPDIR=/home/avifenesh/tmp-lanes \
    tools/local-ci.sh --correctness 2>&1 | tee "$OUT/local-ci.log"
local_ci_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$local_ci_rc" >"$OUT/local-ci.exit"
test "$local_ci_rc" -eq 0
grep -Eq '^kernel-check: ALL GREEN \([0-9]+ cells, [0-9]+ skipped\)$' "$OUT/local-ci.log"
grep -q '^run-spec K=1..8 self-consistency: PASS ' "$OUT/local-ci.log"
grep -q '^serve-smoke: 0 failed$' "$OUT/local-ci.log"

run_gen_gate() {
    local name=$1 model=$2 log=$3 rc
    set +e
    env -u MEMRA_PREFIX_CACHE_MB -u MEMRA_PROMPT_DIR -u MEMRA_SPEC_K \
        MEMRA_NGEN=8 timeout 900 target/release/run-gen "$model" 55 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$rc" >"${log%.log}.exit"
    test "$rc" -eq 0
    grep -q 'MATCH' "$log"
    if grep -q 'MISMATCH' "$log"; then
        echo "FAIL: $name run-gen reported MISMATCH"
        return 1
    fi
    echo "run-gen argmax: MATCH ($name)"
}

run_gen_gate Q27 "$Q27" "$OUT/run-gen-q27.log"
run_gen_gate Q35 "$Q35" "$OUT/run-gen-q35.log"

remaining=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true)
if [ -n "$remaining" ]; then
    echo "FAIL: a CUDA compute process exists after the named gates:"
    echo "$remaining"
    exit 1
fi
echo "GATES_PASS ts=$(date -u +%FT%TZ)"
