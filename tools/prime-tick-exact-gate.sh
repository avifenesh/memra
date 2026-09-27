#!/usr/bin/env bash
# prime-tick-exact-gate — one numeric program per request across the scheduler's prime shapes.
#
# Usage: tools/prime-tick-exact-gate.sh [MODEL] [--canary] [--log PATH]
#
# memra#641 (research/decode-exact-641-20260923). A plain peer primed in tick chunks beside other
# fresh peers produced different greedy bytes than the same prompt primed alone. The engine replay
# (`concat-prime-probe <model> tickshape`) primes peer B the way the non-yielding scheduler did:
# two 1024-row concat batches [A, B, C] (the first fresh, the second carried), C's remaining rows
# in solo tick calls, then B decodes B=1 until step 4 and in a [B, C] wave from there. Every arm
# is teacher-forced on the solo reference's greedy tokens and compared BITWISE (logits, h_seed,
# the full hidden stack, cache digests, every decode step's logits) against `prime_cache(B)` in
# one call. Arms: ref2 (determinism pin), tick, bp, bps, wave. PASS = every arm EXACT.
#
# Registered RED on 9c07b398b: the fresh batch's varlen FA arm (`MEMRA_FA_VL`) attended bf16 of
# the pre-quantization K/V while the solo prime attends the quantized cache view, so bp and bps
# reported `logits bitdiff=248319 ... first_row=Some(0)` and greedy text diverging at token 8.
# The lane's fix deletes that arm; this gate turns green on it.
#
# Liveness: the batched arms call `prime_cache_batch`. If carried-prime.v1 is unqualified that
# entry falls back to individual solo primes and bp/bps pass vacuously, so the gate refuses a log
# that carries the `[rewrite] carried-prime.v1 unqualified` line and unsets MEMRA_REWRITE_BUNDLE.
#
# TEETH: --canary changes the world, not the label: the probe replaces B's first token inside the
# bp/bps batches only. The comparator must then report bp and bps DIFFERS while ref2/tick/wave stay
# EXACT; a canary run where bp or bps is EXACT means the comparator is blind.
#
# GPU: run under the rig lock (fast-gate's cmd rows hold it; standalone, wrap the call in
# `flock /tmp/memra-5090.lock` on the local 5090). Prompts are generated from pinned seeds and
# checked against pinned sha256s, so the ids cannot drift silently. A green run without --log
# removes its scratch dir; a red run keeps the raw log it names.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PROBE=./target/release/concat-prime-probe
CANARY=0
MODEL=""
LOG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --canary) CANARY=1; shift ;;
        --log) LOG="$2"; shift 2 ;;
        -*) echo "prime-tick-exact-gate: unknown arg $1" >&2; exit 2 ;;
        *) MODEL="$1"; shift ;;
    esac
done
MODEL="${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || {
    echo "prime-tick-exact-gate: SKIP (no model at $MODEL)"
    exit 0
}
[ -x "$PROBE" ] || { echo "prime-tick-exact-gate: FAIL (build concat-prime-probe first)"; exit 1; }

WORK=$(mktemp -d /tmp/prime-tick-exact-gate-XXXXXX)
# The ids are regenerated every run; only the raw log outlives the gate (default: in $WORK).
trap 'rm -f "$WORK"/ids-a.json "$WORK"/ids-b.json "$WORK"/ids-c.json; rmdir "$WORK" 2>/dev/null' EXIT
# ids_for(n, seed) as at research/decode-exact-641-20260923/repro641.py: A = peer-cold-a,
# B = peer-cold-b, C = the seed/peer-hit prompt of the prime-fairness gate's #641 run.
if ! python3 - "$WORK" <<'EOF'
import hashlib, json, random, sys
pins = {
    "a": (2048, 5211, "bcf15674488bf08f67daf4586638859ca11388ff89917304f8fdf0beb71848e3"),
    "b": (2048, 5212, "e0d693ebb5113ffb6cb1116d5d3aabaff37d8ee13316df2fa8b80eb6dd20bc02"),
    "c": (4096, 5210, "a3da64743f8a77d938115aa34acbd0cd24131227a7fed9220b3d20675674c37c"),
}
for key, (n, seed, want) in pins.items():
    rng = random.Random(seed)
    text = json.dumps([rng.randrange(1000, 100_000) for _ in range(n)])
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != want:
        sys.exit(f"ids-{key}: sha256 {got} != pinned {want}")
    open(f"{sys.argv[1]}/ids-{key}.json", "w").write(text)
EOF
then
    echo "prime-tick-exact-gate: FAIL (pinned prompt ids did not regenerate; scratch $WORK)"
    exit 1
fi

# A green run without --log removes its scratch; a red one keeps the raw log for the reader.
LOG_KEEP=1
[ -n "$LOG" ] || { LOG="$WORK/tickshape.log"; LOG_KEEP=0; }
green_exit() { [ "$LOG_KEEP" = 1 ] || rm -f "$LOG"; exit 0; }
green_note() { if [ "$LOG_KEEP" = 1 ]; then echo "log $LOG"; else echo "scratch removed"; fi; }
ARGS=(--ids-a "$WORK/ids-a.json" --ids-b "$WORK/ids-b.json" --ids-c "$WORK/ids-c.json"
      --tick 1024 --steps 32 --join 4 --arms "ref,ref2,tick,bp,bps,wave")
[ "$CANARY" = 1 ] && ARGS+=(--canary)
# evidence discipline: the probe writes the raw log, the gate parses the LOG (never the pipe)
env -u MEMRA_REWRITE_BUNDLE "$PROBE" "$MODEL" tickshape "${ARGS[@]}" > "$LOG" 2>&1
rc=$?
grep -E "^(tickshape|arm [a-z0-9]+ (prime|verdict))" "$LOG" | cut -c1-240 | sed 's/^/    /'

fail=0
if grep -q "carried-prime.v1 unqualified" "$LOG"; then
    echo "prime-tick-exact-gate: NOT-LIVE (prime_cache_batch fell back to solo primes; bp/bps"
    echo "  would pass vacuously)"
    fail=1
fi
verdict() { grep -E "^arm $1 verdict: " "$LOG" | head -1 | sed 's/^arm [a-z0-9]* verdict: //'; }
for a in ref ref2 tick wave; do
    v=$(verdict "$a")
    [ "$v" = "EXACT" ] || { echo "prime-tick-exact-gate: arm $a verdict '${v:-missing}' (want EXACT)"; fail=1; }
done
bad_batched=0
for a in bp bps; do
    v=$(verdict "$a")
    [ -n "$v" ] || { echo "prime-tick-exact-gate: arm $a verdict missing"; fail=1; continue; }
    [ "$v" = "EXACT" ] || bad_batched=$((bad_batched + 1))
done

if [ "$CANARY" = 1 ]; then
    if [ $fail -ne 0 ]; then
        echo "prime-tick-exact-gate: CANARY INCONCLUSIVE (the untouched arms or liveness failed;"
        echo "  the run says nothing about the batched comparator; raw log $LOG)"
        exit 1
    fi
    if [ $bad_batched -ne 2 ]; then
        echo "prime-tick-exact-gate: CANARY FAILED (a changed B prompt inside the batch still"
        echo "  read EXACT on $((2 - bad_batched)) batched arm(s): the comparator is blind; log $LOG)"
        exit 1
    fi
    echo "prime-tick-exact-gate: CANARY OK (bp and bps DIFFER with B's batch prompt changed; $(green_note))"
    green_exit
fi

if [ $fail -eq 0 ] && [ $bad_batched -eq 0 ] && [ $rc -eq 0 ] \
    && grep -q "^tickshape verdict: ALL ARMS EXACT" "$LOG"; then
    echo "prime-tick-exact-gate: PASS (every prime shape is bit-identical to the solo prime; $(green_note))"
    green_exit
fi
echo "prime-tick-exact-gate: FAIL rc=$rc (a request primed inside a tick batch or a wave took a"
echo "  different numeric program than the solo prime; memra#641, raw log $LOG)"
exit 1
