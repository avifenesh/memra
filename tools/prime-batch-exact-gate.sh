#!/usr/bin/env bash
# prime-batch-exact-gate: the cross-request batched prime is bit-identical to individual primes.
#
# Usage: tools/prime-batch-exact-gate.sh [MODEL] [--canary] [--logdir DIR]
#
# memra#641 (research/decode-exact-641-20260923). `prime-batch-gate --exact` compares, per
# sequence, `prime_cache_batch` against `prime_cache` bitwise: prefill logits, h_seed, the full
# hidden stack and every teacher-forced decode step's logits. It already existed and was red on the
# 9B at 9c07b398b (`exact-b3-p24 ... "prime-batch-gate: 6 FAIL(s)"`, `seq 0: exact logits diff
# 248320/248320 h_seed diff 4096/4096 hidden diff 98304/98304`), but no battery ran it on the 9B:
# the only fast-gate rows were step35's pbatch35/pbatch35c. The fresh batch's varlen FA arm was
# the cause; its deletion turns every row green. Rows (the lane's battery set):
#   exact-b3-p24      --batch 3 --exact                     (short uneven prompts)
#   exact-b4-p1100    --batch 4 --plen 1100 --exact         (past one 1024-row FA tile)
#   carried-b3-exact  --batch 3 --plen 600 --carried --exact (fresh rows bitwise + carried streams)
#
# Liveness: with an installed rewrite bundle that lacks carried-prime.v1, `prime_cache_batch`
# falls back to individual solo primes and every row passes vacuously, so the gate unsets
# MEMRA_REWRITE_BUNDLE and refuses a log carrying the `carried-prime.v1 unqualified` line.
#
# TEETH: --canary runs exact-b3-p24 with `prime-batch-gate --canary`, which changes seq 0's first
# prompt token inside the batched prime only. seq 0 must then differ bitwise and the run must fail,
# while seqs 1 and 2 stay bit-identical (a batch-mate's prompt must not reach another sequence).
#
# GPU: run under the rig lock (fast-gate's cmd rows and local-ci hold it; standalone, wrap the call
# in `flock /tmp/memra-5090.lock` on the local 5090). Raw logs are kept on a red run; a green run
# removes its scratch unless --logdir names where to keep them.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN=./target/release/prime-batch-gate
CANARY=0
MODEL=""
LOGDIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --canary) CANARY=1; shift ;;
        --logdir) LOGDIR="$2"; shift 2 ;;
        -*) echo "prime-batch-exact-gate: unknown arg $1" >&2; exit 2 ;;
        *) MODEL="$1"; shift ;;
    esac
done
MODEL="${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || {
    echo "prime-batch-exact-gate: SKIP (no model at $MODEL)"
    exit 0
}
[ -x "$BIN" ] || { echo "prime-batch-exact-gate: FAIL (build prime-batch-gate first)"; exit 1; }

KEEP=1
if [ -z "$LOGDIR" ]; then
    LOGDIR=$(mktemp -d /tmp/prime-batch-exact-gate-XXXXXX)
    KEEP=0
fi
mkdir -p "$LOGDIR" || exit 1

# run_row NAME ARGS...: the raw log first, the verdict parsed from the log (never the pipe).
run_row() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    env -u MEMRA_REWRITE_BUNDLE "$BIN" "$MODEL" "$@" > "$log" 2>&1
    local rc=$?
    local last
    last=$(grep -E "^(ALL GREEN|Error)" "$log" | tail -1 | cut -c1-160)
    echo "    $name rc=$rc ${last:-no verdict line}"
    if grep -q "carried-prime.v1 unqualified" "$log"; then
        echo "prime-batch-exact-gate: NOT-LIVE on $name (prime_cache_batch fell back to solo primes;"
        echo "  the row would pass vacuously; log $log)"
        return 3
    fi
    return $rc
}

if [ "$CANARY" = 1 ]; then
    run_row canary-b3-p24 --batch 3 --exact --canary
    rc=$?
    log="$LOGDIR/canary-b3-p24.log"
    [ $rc -eq 3 ] && exit 1
    grep -q "^canary: seq 0 token 0 " "$log" || {
        echo "prime-batch-exact-gate: CANARY INCONCLUSIVE (the binary did not arm the canary; log $log)"
        exit 1
    }
    seq0=$(grep -E "^seq 0: exact logits diff " "$log" | head -1)
    clean=1
    for s in 1 2; do
        grep -Eq "^seq $s: exact logits diff 0/[0-9]+ h_seed diff 0/[0-9]+ hidden diff 0/[0-9]+$" "$log" \
            && grep -q "^seq $s: teacher-forced decode logit diff 0$" "$log" || clean=0
    done
    if [ $clean -ne 1 ]; then
        echo "prime-batch-exact-gate: CANARY FAILED (seq 0's changed prompt moved another sequence's"
        echo "  bits inside the batch; log $log)"
        exit 1
    fi
    if [ $rc -eq 0 ] || [ -z "$seq0" ] || echo "$seq0" | grep -Eq "logits diff 0/"; then
        echo "prime-batch-exact-gate: CANARY FAILED (seq 0's prompt changed inside the batch and the"
        echo "  comparator still read it as exact: it is blind; log $log)"
        exit 1
    fi
    echo "prime-batch-exact-gate: CANARY OK (seq 0 DIFFERS: ${seq0#seq 0: }; seqs 1 and 2 bit-identical)"
    [ $KEEP -eq 0 ] && rm -rf "$LOGDIR"
    exit 0
fi

fail=0
run_row exact-b3-p24 --batch 3 --exact || fail=1
run_row exact-b4-p1100 --batch 4 --plen 1100 --exact || fail=1
run_row carried-b3-exact --batch 3 --plen 600 --carried --exact || fail=1
for name in exact-b3-p24 exact-b4-p1100 carried-b3-exact; do
    grep -q "^ALL GREEN: prime-batch gate" "$LOGDIR/$name.log" || fail=1
done
if [ $fail -eq 0 ]; then
    echo "prime-batch-exact-gate: PASS (every batched prime is bit-identical to its individual prime)"
    [ $KEEP -eq 0 ] && rm -rf "$LOGDIR"
    exit 0
fi
echo "prime-batch-exact-gate: FAIL (a batched prime differs from the individual prime; memra#641,"
echo "  raw logs $LOGDIR)"
exit 1
