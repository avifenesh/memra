#!/usr/bin/env bash
# prime-grid-gate — pins the PRIME-GRID LAW on the qwen hybrid (lane/spec-longctx-20260821,
# the GATES-SMOKE-20260821 B3/B1-fold disposition). Self-gating (`kind=cmd` in
# tools/fast-gate/models.tsv): exit 0 = PASS.
#
#   tools/prime-grid-gate.sh [<model.gguf>] [--prompt <file>] [--aligned 2048,2080]
#                            [--misaligned 2055,3001] [--steps N] [--canary]
#
# THE LAW (measured 2026-08-21, NJ box, q38 27B trunk @ v0.99.0, 24k-token real agentic
# prompt; research/multiturn-cache-20260821/LONGCTX-EXACTNESS-20260821.md): a prompt primed
# as TWO prime_cache calls split at L is BIT-IDENTICAL to the monolithic prime iff
# L % gdn_chunk_size() == 0. Off-grid splits diverge from exactly row L (the chunked WY GDN
# scan segments per call — an off-grid call start shifts the fold grid), and the divergence
# is the LAWFUL near-tie FP class: prefix rows bit-identical, greedy flips only at near-tie
# margins. Under MEMRA_GDN_CHUNKED=0 (sequential scan) every split is EXACT — the mechanism
# pin. serve rounds every boundary it chooses onto the grid (grid_align_boundary), so this
# gate is the engine-side law that fix depends on. Three assertions:
#
#   1. determinism: the monolithic prime run twice is bit-identical (probe hard-stops).
#   2. ALIGNED splits are EXACT — bit-identical logits, zero diverging hidden rows.
#   3. MISALIGNED splits, when they differ, are CONFINED (first diverging hidden row == L,
#      i.e. the shared-prefix rows are bit-identical) and any greedy flip sits at a near-tie
#      margin (margin_ref < 0.5) — wide-margin flips or pre-L divergence = a real defect.
#      A misaligned arm that comes out EXACT also passes: near-tie flips are stochastic and
#      their absence is not a defect (greedy is the instrument, not the product).
#
# TEETH: --canary INJECTS A BREAK (changes the WORLD, not the label): MEMRA_GDN_CHUNK=64
# coarsens the fold grid so the 32-aligned-but-not-64-aligned split (2080 in the default
# set — keep one such split when overriding --aligned) stops being aligned, and assertion 2
# must FAIL. A canary run where the gate still passes means the gate cannot detect the
# mechanism and must itself be fixed.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PROBE=./target/release/concat-prime-probe
MODEL=""
PROMPT=research/chunk-invariance-20260805/prompt-pp6257.txt
ALIGNED=2048,2080
MISALIGNED=2055,3001
STEPS=24
CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --prompt)     PROMPT="$2"; shift 2 ;;
        --aligned)    ALIGNED="$2"; shift 2 ;;
        --misaligned) MISALIGNED="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --canary)     CANARY=1; shift ;;
        -*) echo "prime-grid-gate: unknown arg $1" >&2; exit 2 ;;
        *)  MODEL="$1"; shift ;;
    esac
done
MODEL="${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || [ -d "$MODEL" ] || {
    echo "prime-grid-gate: SKIP (no model or safetensors dir at $MODEL)"
    exit 0
}
[ -x "$PROBE" ] || { echo "prime-grid-gate: FAIL (build concat-prime-probe first)"; exit 1; }
[ -f "$PROMPT" ] || { echo "prime-grid-gate: FAIL (missing pinned prompt $PROMPT)"; exit 1; }

LOG=$(mktemp /tmp/prime-grid-gate-XXXXXX.log)
ENVX=()
[ "$CANARY" = 1 ] && ENVX=("MEMRA_GDN_CHUNK=64")
# evidence discipline: tee the raw log, parse the LOG (never the pipe)
env -u MEMRA_GDN_CHUNK -u MEMRA_GDN_CHUNKED "${ENVX[@]}" \
    "$PROBE" "$MODEL" primepath --prompt-a "@$PROMPT" \
    --splits "$ALIGNED,$MISALIGNED" --steps "$STEPS" > "$LOG" 2>&1
rc=$?
grep -E "^(primepath:|arm |verdict )" "$LOG" | sed 's/^/    /'
if [ $rc -ne 0 ]; then
    echo "prime-grid-gate: FAIL (probe rc=$rc — the mono2 determinism pin hard-stops on"
    echo "  nondeterminism; raw log $LOG)"
    exit 1
fi

fail=0
for l in $(echo "$ALIGNED" | tr ',' ' '); do
    if ! grep -qE "^arm sp$l: logits EXACT \| rows_diff 0/" "$LOG"; then
        echo "prime-grid-gate: ALIGNED split sp$l is NOT bit-identical to the monolithic prime"
        fail=1
    fi
done
for l in $(echo "$MISALIGNED" | tr ',' ' '); do
    row=$(grep -E "^arm sp$l: " "$LOG" || true)
    [ -n "$row" ] || { echo "prime-grid-gate: missing arm sp$l"; fail=1; continue; }
    if echo "$row" | grep -q "logits EXACT"; then
        continue # near-tie flips are stochastic; exact misaligned arms are lawful too
    fi
    first=$(echo "$row" | sed -n 's/.*rows_diff [0-9]*\/[0-9]* first \([0-9-]*\).*/\1/p')
    if [ "$first" != "$l" ]; then
        echo "prime-grid-gate: sp$l diverges BEFORE the split row ($first < $l) — not the"
        echo "  confined lawful class; a shared-prefix row moved. Real defect."
        fail=1
    fi
    m=$(echo "$row" | sed -n 's/.*flip step [0-9]*: margin_ref \([0-9.e-]*\).*/\1/p')
    if [ -n "$m" ] && ! awk -v m="$m" 'BEGIN { exit !(m < 0.5) }'; then
        echo "prime-grid-gate: sp$l greedy flip at WIDE margin $m (>= 0.5) — not a near-tie;"
        echo "  the perturbation at the contending ids exceeds the lawful class. Real defect."
        fail=1
    fi
done

if [ "$CANARY" = 1 ]; then
    if [ $fail -eq 0 ]; then
        echo "prime-grid-gate: CANARY UNEXPECTEDLY PASSED — coarsening the fold grid"
        echo "  (MEMRA_GDN_CHUNK=64) did not break the aligned-split assertion, so this gate"
        echo "  cannot detect the mechanism. FIX THE GATE (keep one 32-not-64-aligned split"
        echo "  in --aligned). (log $LOG)"
        exit 1
    fi
    echo "prime-grid-gate: PASS (canary broke the assertion as required — gate has teeth; log $LOG)"
    exit 0
fi
if [ $fail -eq 0 ]; then
    echo "prime-grid-gate: PASS (raw log $LOG)"
    exit 0
fi
echo "prime-grid-gate: FAIL — the prime-grid law moved. serve's grid_align_boundary and the"
echo "  prefix-cache capture alignment DEPEND on aligned splits being bit-identical; re-root-"
echo "  cause against research/multiturn-cache-20260821/LONGCTX-EXACTNESS-20260821.md before"
echo "  touching this gate. raw log: $LOG"
exit 1
