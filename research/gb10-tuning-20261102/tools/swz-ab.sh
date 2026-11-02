#!/usr/bin/env bash
# Runs ON spark-cb33 under the caller's lease: W4A8 MMQ rasterization A/B on GB10.
#   BIN=~/memra-run/bin-<sha12> GS="0 4 8 16" TOKENS=8192 ROUNDS=2 swz-ab.sh
# Each round runs every arm in order then in reverse (balanced, both orders); each arm is one
# process with one warmup and REPS timed monolithic pp-only forwards. Last-row prefill logits are
# dumped per arm and byte-compared against G=0.
set -u
BIN=${BIN:?bin}; GS=${GS:-0 4 8 16}; TOK=${TOKENS:-8192}; ROUNDS=${ROUNDS:-2}
M=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
O=$HOME/memra-run/swz; mkdir -p "$O"
P=$HOME/memra-run/prof/prompt-v2-$TOK.txt
[ -s "$P" ] || { echo "missing prompt $P (run profile-prefill.sh once)"; exit 2; }
export MEMRA_PROMPT_FILE=$P MEMRA_CTX=$((TOK*2)) MEMRA_PP_ONLY=1 MEMRA_PP_WARMUP=1 MEMRA_PP_REPS=${REPS:-3}
sha256sum "$BIN/run-gen"
nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.draw --format=csv,noheader
rev=$(echo $GS | tr ' ' '\n' | tac | tr '\n' ' ')
for r in $(seq 1 "$ROUNDS"); do
  for g in $GS $rev; do
    out=$(MEMRA_MMQ_SWIZZLE=$g MEMRA_PP_LOGITS=$O/logits-$TOK-g$g.bin "$BIN/run-gen" "$M" 2>&1)
    echo "round=$r g=$g $(echo "$out" | grep -E 'pp-only rep|pp-only MEDIAN' | tr '\n' ' ' | cut -c1-300)"
  done
done
for g in $GS; do cmp -s "$O/logits-$TOK-g0.bin" "$O/logits-$TOK-g$g.bin" && echo "g=$g logits IDENTICAL to g=0" || echo "g=$g logits DIFFER"; done
sha256sum "$O"/logits-$TOK-g*.bin
