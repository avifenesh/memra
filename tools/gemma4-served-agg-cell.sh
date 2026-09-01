#!/usr/bin/env bash
# gemma4 SERVED AGGREGATE cells (lane/gemma-batched, 2026-08-16): seam-on vs seam-off,
#
# Q8RP note: the 2026-08-17 mirror regression (NVFP4 prefill NaN on 96GB boots) is
# FIXED at lane/gemma-pnfold abf155e8 (build_q4_rp_swap qtype guard) — no pin needed
# at or after that commit; the boot output-sample gate is the standing guard.
# c1/c4/c8/c16 through the real HTTP surface, INTERLEAVED xN per the A/B law (box-clock
# drift invalidates cross-run comparisons — never run all-A then all-B).
#
# One rep = boot seam-off -> measure all concurrencies -> boot seam-on -> measure all
# concurrencies. Reps alternate boot order (off-first, on-first, ...) so warmup/thermal
# order effects cancel. Emits one JSONL point per (rep, seam, concurrency) to $OUT.
#
# Usage: CUDA_VISIBLE_DEVICES=0 tools/gemma4-served-agg-cell.sh <model.gguf> <out.jsonl> [reps]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?model.gguf}"
OUT="${2:?out.jsonl}"
REPS="${3:-5}"
CONCS=(${CONCS_OVERRIDE:-1 4 8 16})
ADDR=127.0.0.1:8183
BASE=http://$ADDR
LOGDIR="$(dirname "$OUT")"
mkdir -p "$LOGDIR"

start_server() {  # $1 = seam value or "unset" (the shipping default), $2 = log
  local envargs=()
  if [ "$1" != "unset" ]; then envargs=(MEMRA_GEMMA4_BATCH="$1"); fi
  env "${envargs[@]}" MEMRA_COMPAT=openai MEMRA_MODELS="g4=$MODEL" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 240); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; tail:"; tail -8 "$2"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }
trap stop_server EXIT

measure_side() {  # $1 = seam ("unset" = shipping default = batched; "0" = eager kill switch), $2 = rep
  local seam=$1 rep=$2 tag log
  tag=$([ "$seam" = "0" ] && echo off || echo on)
  log="$LOGDIR/server-$tag-r$rep.log"
  start_server "$seam" "$log" || return 1
  # route sanity (refuse-on-ambiguity): the boot notice must match the seam.
  if [ "$seam" = "0" ]; then
    grep -q "EAGER-ONLY serving" "$log" || { echo "AMBIGUOUS: kill-switch boot lacks EAGER-ONLY notice"; return 1; }
  else
    grep -q "BATCHED DECODE (gemma4 dense arm, default-on" "$log" || { echo "AMBIGUOUS: batched-side boot lacks BATCHED notice"; return 1; }
  fi
  for c in "${CONCS[@]}"; do
    python3 tools/load-serve.py --base $BASE --model g4 --concurrency "$c" \
      --requests $((c * 3)) --max-tokens 128 --warmup 1 --timeout 900 \
      --label "g4-$tag-c$c-r$rep" --out "$OUT" || echo "load point c=$c $tag r$rep FAILED"
  done
  # batched-side runs must show the batched walk engaged at least once past c1.
  if [ "$seam" != "0" ] && ! grep -q "\[gemma4-batch\] first B>1" "$log"; then
    echo "AMBIGUOUS: seam-on rep $rep never formed a B>1 chunk"
  fi
  stop_server
}

SEAM_A="${SEAM_A:-0}"       # eager kill switch
SEAM_B="${SEAM_B:-unset}"   # shipping default (batched)
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then order="$SEAM_A $SEAM_B"; else order="$SEAM_B $SEAM_A"; fi
  for seam in $order; do
    echo "== rep $rep seam=$seam =="
    measure_side "$seam" "$rep" || exit 1
  done
done
echo "cells complete -> $OUT"
