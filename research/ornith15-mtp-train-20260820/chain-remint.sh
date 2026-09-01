#!/usr/bin/env bash
# GGUF remint with the v2-trained MTP head: patch BF16 GGUF -> NVFP4 mint (same
# sealed recipe as the published artifact) -> masked draft v3 -> GGUF gates ->
# serve-level A/B v2-gguf vs published v1 gguf.
set -uo pipefail
cd "$HOME/models/ornith15"
BINS=$HOME/memra-src/target/release
Q=$HOME/llama-nvfp4/build/bin/llama-quantize
BF16=gguf-src/Ornith-1.5-35B-BF16.gguf
OUT=Ornith-1.5-35B-A3B-NVFP4-MTP-v2.gguf
DRAFT=mtp-Ornith-1.5-35B-A3B-NVFP4-frspec-owngen32768-v2.gguf
mkdir -p gates-remint

echo "== 1. patch blk.40 with v2 head =="
python3 mtp-train/patch_gguf_mtp.py --gguf "$BF16" \
  --mtp mtp-train/train-v2-out/mtp-trained-epoch2.safetensors > gates-remint/patch.log 2>&1
grep -q "GGUF PATCH DONE" gates-remint/patch.log || { echo "FATAL: gguf patch"; tail -5 gates-remint/patch.log; exit 1; }
tail -2 gates-remint/patch.log

echo "== 2. NVFP4 mint (sealed recipe) =="
nice -n 5 "$Q" --output-tensor-type q5_k --token-embedding-type q5_k \
  "$BF16" "$OUT" NVFP4 > gates-remint/quantize.log 2>&1 || { echo "FATAL: quantize"; tail -3 gates-remint/quantize.log; exit 1; }
sha256sum "$OUT" | tee "$OUT.sha256"

echo "== 3. masked draft v3 (own-gen ranks, hqmtp order) =="
MEMRA_GGUFPY=$HOME/llama-nvfp4/gguf-py MEMRA_QUANTIZE=$Q \
  bash "$HOME/memra-src/tools/make-trimmed-draft.sh" "$OUT" \
  ornith15-ranks-owngen-32768.txt "$DRAFT" 32768 > gates-remint/draft.log 2>&1 \
  || { echo "FATAL: draft"; tail -5 gates-remint/draft.log; exit 1; }
sha256sum "$DRAFT" | tee "$DRAFT.sha256"

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FATAL: gpu lock"; exit 1; }
export CUDA_VISIBLE_DEVICES=0

echo "== 4. run-gen chat probes =="
: > gates-remint/run-gen.log
for prm in "Explain the difference between TCP and UDP in two sentences." "Write a Python function that reverses a linked list."; do
  echo "=== PROMPT: $prm" >> gates-remint/run-gen.log
  MEMRA_CHAT=1 "$BINS/run-gen" "$OUT" --prompt "$prm" >> gates-remint/run-gen.log 2>&1
done

echo "== 5. run-spec K=1..8 self-consistency =="
MEMRA_FAST=1 "$BINS/run-spec" "$OUT" > gates-remint/run-spec.log 2>&1 \
  && echo "run-spec PASS" || { echo "run-spec FAIL"; tail -5 gates-remint/run-spec.log; }

echo "== 6. serve A/B: v1 gguf vs v2 gguf (embedded heads) + v2 masked draft =="
RC=0
python3 mtp-train/ab_head.py \
  --arms "v1gguf=$HOME/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-MTP.gguf,v2gguf=$HOME/models/ornith15/$OUT" \
  --out gates-remint/ab-gguf.jsonl > gates-remint/ab-gguf.log 2>&1 || RC=$?
tail -2 gates-remint/ab-gguf.log
echo "REMINT-CHAIN DONE rc=$RC"
exit "$RC"
