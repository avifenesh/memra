#!/usr/bin/env bash
# Runs ON gb10-b: W4A8 (default) vs F8F4 (MEMRA_MMQ_F8F4=1) prefill rate on GB10, both orders,
# plus last-row logits for each arm. Usage: BIN=... TOKENS=8192 f8f4-pp.sh
set -u
BIN=${BIN:?bin}; TOK=${TOKENS:-8192}
M=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
O=$HOME/memra-run/f8f4; mkdir -p "$O"
export MEMRA_PROMPT_FILE=$HOME/memra-run/prof/prompt-v2-$TOK.txt MEMRA_CTX=$((TOK*2)) MEMRA_PP_ONLY=1 MEMRA_PP_WARMUP=1 MEMRA_PP_REPS=3
for arm in 0 1 1 0; do
  out=$(MEMRA_MMQ_F8F4=$arm MEMRA_PP_LOGITS=$O/logits-$TOK-f8f4$arm.bin "$BIN/run-gen" "$M" 2>&1)
  echo "f8f4=$arm $(echo "$out" | grep -E 'pp-only MEDIAN|f8f4|F8F4' | tr '\n' ' ' | cut -c1-200)"
done
cmp -s "$O/logits-$TOK-f8f40.bin" "$O/logits-$TOK-f8f41.bin" && echo "logits IDENTICAL" || echo "logits DIFFER (expected: e4m3 activations are a different numeric program)"
