#!/usr/bin/env bash
# prime-split-gate — the PRIME PP schedule gate (lane/pp-leverb +
# lane/cx-pipeline-prime, 2026-08-08): unsplit, serial split, and pipelined split must be
# BIT-IDENTICAL, with both the split and the chunk-overlap schedules provably LIVE.
# Self-gating (`kind=cmd` in tools/fast-gate/models.tsv):
# exit 0 = PASS.
#
#   tools/prime-split-gate.sh [<model.gguf>] [--devices 0,1] [--stages 2] [--chunks auto,513]
#                             [--steps 8] [--prompts <f>] [--canary]
#
# WHY THIS GATE EXISTS (research/pp-leverb-20260807/PROGRESS.md): the prime path has NO pp
# stage split — under MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 every prime chunk walks all 45
# layers on dev0, peer-reading stage-1 trunk weights (22% of the pp4096 wall) while dev1 runs
# ZERO kernels (anatomy receipt: kernels per device = [(0, 2337323, 87.6s)]). Prime keeps NO
# refuse_unsplit_if_remote — its unsplit walk is a 22% amortized tax, not the decode 28x
# cliff, and it is precisely this gate's REFERENCE arm (MEMRA_PRIME_PP=0).
#
# SCHEDULE CONTRACT: the unsplit and serial arms use the fixed rollback schedule; the
# pipeline arm uses the dynamic naked-auto schedule. Their returned tensors and primed-cache
# continuation must remain bit-identical. Explicit --chunks values remain fixed in all arms.
#
# LIVENESS TEETH: the fixed-serial and dynamic-pipeline arms must advance
# PRIME_SPLIT_CHUNKS; only the pipeline arm may advance PRIME_PIPE_OVERLAPS, by at least
# chunks-1. --canary passes --force-serial-pipe: the pipeline arm remains a real stage
# split with dynamic boundaries but takes MEMRA_PRIME_PIPE=0. Bits still agree, split
# liveness still passes, and ONLY the overlap assertion must turn RED.
#
# NEEDS 2-4 GPUs with P2P. On a single-GPU rig (the local 5090) it SKIPs:
# a same-device "split" exercises the seam but not the placement this lever exists for; the
# box battery is the authority (CLAUDE.md: CI is compile-only, the battery is the real gate).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PROBE=./target/release/concat-prime-probe
MODEL=""
DEVICES=0,1
STAGES=2
CHUNKS=auto,513
STEPS=8
PROMPTS=""
CANARY=0
need_value() {
    [ "$#" -ge 2 ] || { echo "prime-split-gate: missing value for $1" >&2; exit 2; }
}
while [ $# -gt 0 ]; do
    case "$1" in
        --devices) need_value "$@"; DEVICES="$2"; shift 2 ;;
        --stages)  need_value "$@"; STAGES="$2"; shift 2 ;;
        --chunks)  need_value "$@"; CHUNKS="$2"; shift 2 ;;
        --steps)   need_value "$@"; STEPS="$2"; shift 2 ;;
        --prompts) need_value "$@"; PROMPTS="$2"; shift 2 ;;
        --canary)  CANARY=1; shift ;;
        -*) echo "prime-split-gate: unknown arg $1" >&2; exit 2 ;;
        *)  MODEL="$1"; shift ;;
    esac
done
case "$STAGES" in
    2|3|4) ;;
    *) echo "prime-split-gate: FAIL (--stages must be exactly 2, 3, or 4)" >&2; exit 2 ;;
esac
if [[ ! "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "prime-split-gate: FAIL (--devices must be a comma-separated list of numeric ordinals)" >&2
    exit 2
fi
IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
if [ "${#DEVICE_LIST[@]}" -ne "$STAGES" ]; then
    echo "prime-split-gate: FAIL (--devices count must equal --stages)" >&2
    exit 2
fi
# Default model = the launch SKU (the placement this lever serves); resolves like chunkinv35.
if [ -z "$MODEL" ]; then
    for cand in "${MEMRA_STEP37_GGUF:-}" \
        "$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        /data/ai-ml/hf-models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf; do
        [ -n "$cand" ] && [ -f "$cand" ] && { MODEL="$cand"; break; }
    done
    [ -z "$MODEL" ] && { echo "prime-split-gate: SKIP (no Step-3.7-Flash artifact; set MEMRA_STEP37_GGUF)"; exit 0; }
fi
[ -f "$MODEL" ] || { echo "prime-split-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$PROBE" ] || { echo "prime-split-gate: FAIL (build concat-prime-probe first)"; exit 1; }
# Distinct-device count must cover the placement (single-GPU rigs SKIP — see header).
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
NDEV=$(echo "$DEVICES" | tr ',' '\n' | sort -u | wc -l)
MAXDEV=$(echo "$DEVICES" | tr ',' '\n' | sort -n | tail -1)
if [ "$NGPU" -le "$MAXDEV" ] || [ "$NDEV" -ne "$STAGES" ]; then
    echo "prime-split-gate: SKIP (needs the multi-GPU placement $DEVICES; $NGPU GPU(s) visible)"
    exit 0
fi
# Prompt must exercise both the naked auto geometry and the fixed stress chunk.
PROMPTS="${PROMPTS:-research/chunk-invariance-20260805/prompt-pp6257.txt}"
[ -f "$PROMPTS" ] || { echo "prime-split-gate: FAIL (missing pinned prompt $PROMPTS)"; exit 1; }

EXTRA=()
[ "$CANARY" = 1 ] && EXTRA=(--force-serial-pipe)
LOG=$(mktemp /tmp/prime-split-gate-XXXXXX.log)
# evidence discipline: tee the raw log, parse the LOG (never the pipe)
WAVE_ENV=()
if [ "$STAGES" -gt 2 ]; then
    WAVE_ENV=(MEMRA_PP_WAVE=1 MEMRA_PP_OVERLAP=1)
fi
env "${WAVE_ENV[@]}" "MEMRA_PP_STAGES=$STAGES" "MEMRA_PP_DEVICES=$DEVICES" \
    "$PROBE" "$MODEL" ppsplit --prompt-a "@$PROMPTS" \
    --chunks "$CHUNKS" --steps "$STEPS" "${EXTRA[@]}" > "$LOG" 2>&1
rc=$?
grep -E "^ppsplit|^  chunk" "$LOG" | sed 's/^/    /'
if [ "$CANARY" = 1 ]; then
    # CANARY EVIDENCE CONTRACT (GATE-INTEGRITY-20260819 A-10, fixed 2026-08-19).
    # `rc -ne 0` was the whole test, and it cannot tell the injected defect from the probe
    # dying: a panic (101), a usage error from a renamed flag (2), a DOOR-SHUT refusal because
    # MEMRA_PP_STAGES was exported too late, or the model failing to load all read as "PASS
    # (serial-pipeline canary broke overlap liveness as required)".
    #
    # The header states exactly which assertion the canary may break: "Bits still agree, split
    # liveness still passes, and ONLY the overlap assertion must turn RED." concat_prime_probe
    # emits that as the per-chunk status word `*** PIPE-NOT-LIVE (serial split replayed)`, and
    # `*** MISMATCH` / `*** SPLIT-NOT-LIVE` for the other two. So assert the specific word, and
    # refuse the others by name rather than counting them as success.
    if [ $rc -eq 0 ]; then
        echo "prime-split-gate: CANARY UNEXPECTEDLY MATCHED — forcing the pipeline arm serial did not"
        echo "  flip the verdict, so overlap liveness cannot detect the mechanism. FIX THE GATE. (log $LOG)"
        exit 1
    fi
    if grep -q 'ppsplit verdict: \*\*\* DOOR-SHUT' "$LOG"; then
        echo "prime-split-gate: CANARY INCONCLUSIVE — the PP door was shut (MEMRA_PP_STAGES must be"
        echo "  exported before load), so no split ran at all and no assertion was tested. (log $LOG)"
        exit 1
    fi
    if ! grep -q 'ppsplit verdict: \*\*\* RED' "$LOG"; then
        echo "prime-split-gate: CANARY INCONCLUSIVE — rc=$rc but the probe never printed its own"
        echo "  'ppsplit verdict: *** RED'. It died before asserting (panic, usage error, load"
        echo "  failure), which says nothing about the overlap-liveness assertion. (log $LOG)"
        exit 1
    fi
    if grep -qE '\*\*\* (MISMATCH|SPLIT-NOT-LIVE)' "$LOG"; then
        echo "prime-split-gate: CANARY FAILED — the RED came from"
        echo "  $(grep -Eom1 '\*\*\* (MISMATCH|SPLIT-NOT-LIVE)[^|]*' "$LOG"), not from overlap liveness."
        echo "  --force-serial-pipe must keep bits identical and the split live; if it does not,"
        echo "  the canary is injecting a DIFFERENT defect than the one it claims to test. (log $LOG)"
        exit 1
    fi
    if grep -q 'PIPE-NOT-LIVE' "$LOG"; then
        echo "prime-split-gate: PASS (serial-pipeline canary turned exactly the overlap assertion"
        echo "  RED — PIPE-NOT-LIVE, bits identical, split still live; log $LOG)"
        exit 0
    fi
    echo "prime-split-gate: CANARY INCONCLUSIVE — RED verdict with no PIPE-NOT-LIVE chunk status."
    echo "  The overlap assertion is not what failed. (log $LOG)"
    exit 1
fi
if [ $rc -eq 0 ]; then
    echo "prime-split-gate: PASS (unsplit/serial/pipeline bit-identical + live; raw log $LOG)"
    exit 0
fi
echo "prime-split-gate: FAIL rc=$rc — split/pipeline absent, not live, or not bit-identical (log $LOG)"
exit 1
