#!/usr/bin/env bash
# step35-prime-batch-gate — exact cross-request prime gate over the real PP-2 placement.
#
# Assertions:
#   1. B=2 uneven prompts beyond the 512-token SWA window produce bit-identical logits,
#      h_seed, full hidden stacks, and teacher-forced decode logits vs serial primes.
#   2. The dedicated step35 batch path ran.
#   3. Its PP stage split ran; an unsplit whole-trunk walk is a vacuous correctness pass.
#
# Registered RED (lane/cx-prime-batch, 2026-08-08): prime_cache_batch currently refuses
# step35, so the gate exits nonzero before any liveness counter advances.
#
# --canary sets MEMRA_STEP35_PRIME_BATCH=0. Once the mechanism lands, this must restore the
# refusal and break the naked gate. While the gate is registered-red, both arms are red by
# construction; the canary becomes load-bearing with the implementation commit.
#
# CANARY EVIDENCE CONTRACT (GATE-INTEGRITY-20260819 A-10, fixed 2026-08-19). The canary arm
# used to accept ANY nonzero exit as teeth. `RC` here can be 75 from `flock -w 3600 || exit 75`
# — i.e. "the lock was busy, the gate never ran" reading as "the comparator correctly rejected
# the injected defect". It can also be 2 from a renamed flag's usage error, or 1 from the
# binary failing to load the model. A teeth-proof that cannot tell "the comparator rejected the
# defect" from "nothing ran" proves nothing about the naked arm.
# The canary now requires the COMPARATOR'S OWN words, captured to its own file (the correct
# form already in-tree: iso-gap-gate.sh greps CANARY-OK; argmax-margin-gate.sh reads the
# comparator's `bad` counter).
set -uo pipefail
cd "$(dirname "$0")/.."

CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --canary) CANARY=1; shift ;;
        *) echo "step35-prime-batch-gate: unknown arg $1" >&2; exit 2 ;;
    esac
done

MODEL="${MEMRA_STEP37_GGUF:-}"
if [ -z "$MODEL" ]; then
    for cand in \
        "$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        "/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        "/data/models/step37/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        "/data/models/step37/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"; do
        [ -f "$cand" ] && { MODEL="$cand"; break; }
    done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || {
    echo "step35-prime-batch-gate: SKIP (no Step-3.7-Flash artifact; set MEMRA_STEP37_GGUF)"
    exit 0
}

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU" -ge 2 ] || {
    echo "step35-prime-batch-gate: SKIP (needs the two-GPU PP placement, have $NGPU)"
    exit 0
}

BIN=./target/release/prime-batch-gate
[ -x "$BIN" ] || {
    echo "step35-prime-batch-gate: FAIL (no $BIN — build release first)"
    exit 1
}

TS=$(date -u +%Y%m%dT%H%M%SZ)
TAG=$([ "$CANARY" = 1 ] && echo canary || echo naked)
D=research/primebatch-20260808/raw
mkdir -p "$D"
LOG=$D/primebatch35-$TAG-$TS.log
# The COMPARATOR's own output, on its own file. Reading it back out of $LOG would race the
# `tee` in the process substitution below (its buffer is not ours to flush), and a canary that
# reads a half-written log is the same class of bug as one that reads only an exit code.
CMP=$D/primebatch35-cmp-$TAG-$TS.log
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-gpu.lock}
exec > >(tee "$LOG") 2>&1

echo "=== step35-prime-batch-gate tag=$TAG ts=$TS model=$MODEL lock=$GPU_LOCK ==="
RC=1
(
    flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
    echo "lock acquired $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    CANARY_ENV=()
    [ "$CANARY" = 1 ] && CANARY_ENV=(MEMRA_STEP35_PRIME_BATCH=0)
    # pipefail is set at the top, so `$?` here is the BINARY's status, not tee's.
    env "${CANARY_ENV[@]}" \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
        "$BIN" "$MODEL" --batch 2 --plen 520 --steps 4 --exact --require-pp-split 2>&1 \
        | tee "$CMP"
    rc=$?
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
) 9>"$GPU_LOCK"
RC=$?

if [ "$CANARY" = 1 ]; then
    if [ "$RC" -eq 0 ]; then
        echo "step35-prime-batch-gate: CANARY FAILED (gate passed with the rollback seam on)"
        exit 1
    fi
    if [ "$RC" -eq 75 ]; then
        echo "step35-prime-batch-gate: CANARY INCONCLUSIVE — rc=75 is the flock -w 3600"
        echo "  timeout, i.e. the GPU lock ($GPU_LOCK) was held and the comparator never ran."
        echo "  'The lock was busy' is not 'the rollback broke exactness'. Re-run when free."
        exit 1
    fi
    # The comparator's own verdict words. prime_batch_gate.rs emits `MISMATCH` per diverging
    # sequence, `batched-prime liveness: ... NOT-LIVE` when the step35 batch or its PP split
    # did not run, and `prime-batch-gate: <n> FAIL(s)` as its final error.
    if [ -s "$CMP" ] && grep -Eq 'prime-batch-gate: [0-9]+ FAIL\(s\)|MISMATCH|batched-prime liveness:.*NOT-LIVE' "$CMP"; then
        echo "step35-prime-batch-gate: CANARY OK (rollback broke exact+live as required:" \
             "$(grep -Eom1 'prime-batch-gate: [0-9]+ FAIL\(s\)|MISMATCH|NOT-LIVE' "$CMP"))"
        exit 0
    fi
    echo "step35-prime-batch-gate: CANARY INCONCLUSIVE — rc=$RC but the comparator printed no"
    echo "  MISMATCH, no NOT-LIVE liveness verdict, and no 'N FAIL(s)' tally. The run died"
    echo "  BEFORE asserting anything (missing model, refusal at load, usage error), so this"
    echo "  says nothing about whether the naked arm has teeth. Comparator output: $CMP"
    exit 1
fi

if [ "$RC" -eq 0 ]; then
    echo "step35-prime-batch-gate: PASS (serial identity + live PP-split batch)"
else
    echo "step35-prime-batch-gate: FAIL rc=$RC (refusal, mismatch, or split not live)"
fi
exit "$RC"
