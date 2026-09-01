#!/usr/bin/env bash
# ONE DEPTH RUNG. Wraps probe.py so each rung's receipt carries not just the timing but the
# SERVER-SIDE evidence produced by that exact request: the [glm5-acc] acceptance lines, any
# [glm5-phase]/[glm5-phase-v] trace lines, the per-card VRAM after, and an error census.
# Acceptance is read from the server's own log window for the request, never inferred from
# tok/s (counters prove engagement; they never price it - tpd-battery trap).
# usage: rung.sh <boot-name> <label> <chars> <max_tokens> <greedy|vendor> <outdir>
set -uo pipefail
OUT=/root/out-1m
BOOT=$1; LABEL=$2; CHARS=$3; MAXTOK=$4; MODE=$5; DEST=$6
LOG=$OUT/logs/boot-$BOOT.log
C=/root/corpus-1m/corpus-1m.txt
mkdir -p "$DEST"
BEFORE=$(wc -l < "$LOG")
echo "### RUNG $LABEL mode=$MODE chars=$CHARS maxtok=$MAXTOK boot=$BOOT start=$(date -u +%FT%TZ)"
S=$SECONDS
python3 "$OUT/probe.py" "$LABEL" "$C" "$CHARS" "$MAXTOK" "$MODE" "$DEST/$LABEL-$MODE.json"
rc=$?
echo "  rung wall $((SECONDS-S))s rc=$rc end=$(date -u +%FT%TZ)"
# server-side evidence for THIS request only (the log window it produced)
tail -n +$((BEFORE+1)) "$LOG" > "$DEST/$LABEL-$MODE.serverlog"
{
  echo "--- [glm5-acc] acceptance lines for this request ---"
  grep -E "\[glm5-acc\]" "$DEST/$LABEL-$MODE.serverlog" | tail -20
  echo "--- [glm5-phase] / [glm5-phase-v] (empty unless trace armed) ---"
  grep -E "\[glm5-phase" "$DEST/$LABEL-$MODE.serverlog" | tail -20
  echo "--- error census (must be 0) ---"
  grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error|OUT_OF_MEMORY|\[admit-oom\]" "$DEST/$LABEL-$MODE.serverlog"
  echo "--- errors, if any ---"
  grep -E "out.of.memory|panicked|CUDA_ERROR|engine-error|OUT_OF_MEMORY|\[admit-oom\]|\[admission\] request cost" "$DEST/$LABEL-$MODE.serverlog" | head -8
} > "$DEST/$LABEL-$MODE.evidence"
cat "$DEST/$LABEL-$MODE.evidence"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$DEST/$LABEL-$MODE.vram"
echo "  vram after: $(paste -sd' | ' < "$DEST/$LABEL-$MODE.vram")"
exit $rc
