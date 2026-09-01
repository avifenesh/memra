#!/usr/bin/env bash
# Phase 1 on B200: corpus build + response regeneration by the FP8 target (temp 0, both modes).
# All 8 cards serve replicas. Idempotent; regen supports --resume. Logs: ~/regen.log
set -ux
exec >> /home/ubuntu/regen.log 2>&1
export PATH="$HOME/.local/bin:$PATH"
PYE=/scratch/venvs/eval/bin/python
SF=/scratch/repos/SpecForge
OUT=/scratch/corpus
mkdir -p $OUT/regen /scratch/receipts/regen

# 1. own-corpus prompts
[ -s $OUT/own-prompts.jsonl ] || $PYE /home/ubuntu/sessions_to_corpus.py \
  --corpus-root /scratch/corpus/sessions --out $OUT/own-prompts.jsonl

# 2. perfectblend control subset (30K)
if [ ! -s $OUT/perfectblend-30k.jsonl ]; then
  cd $SF && VIRTUAL_ENV=/scratch/venvs/eval /scratch/venvs/eval/bin/python scripts/prepare_data.py \
    --dataset perfectblend --output-dir $OUT/pb-full || true
  PB=$(find $OUT/pb-full -name '*.jsonl' | head -1)
  [ -n "$PB" ] && head -30000 "$PB" > $OUT/perfectblend-30k.jsonl
fi

# 3. eight FP8 replica servers (plain, no spec)
for g in 0 1 2 3 4 5 6 7; do
  tmux has-session -t srv$g 2>/dev/null || tmux new-session -d -s srv$g \
    "CUDA_VISIBLE_DEVICES=$g $PYE -m sglang.launch_server --trust-remote-code \
     --model-path /scratch/models/qwen38-27b-fp8 --tp-size 1 \
     --mamba-scheduler-strategy extra_buffer --attention-backend flashinfer \
     --mem-fraction-static 0.85 --port 3000$g 2>&1 | tee /home/ubuntu/srv$g.log"
done
for g in 0 1 2 3 4 5 6 7; do
  until curl -s --max-time 3 localhost:3000$g/health >/dev/null; do sleep 15; done
done
SRV="localhost:30000 localhost:30001 localhost:30002 localhost:30003 localhost:30004 localhost:30005 localhost:30006 localhost:30007"

# 4. regenerate: own corpus both modes, perfectblend both modes (temp 0, resume-safe)
REGEN="$PYE $SF/scripts/regenerate_train_data.py --model /scratch/models/qwen38-27b-fp8 \
  --temperature 0 --max-tokens 8192 --concurrency 48 --resume"
$REGEN --reasoning save    --input-file-path $OUT/own-prompts.jsonl      --output-file-path $OUT/regen/own-think.jsonl    --server-address $SRV || true
$REGEN --reasoning disable --input-file-path $OUT/own-prompts.jsonl      --output-file-path $OUT/regen/own-nothink.jsonl  --server-address $SRV || true
$REGEN --reasoning save    --input-file-path $OUT/perfectblend-30k.jsonl --output-file-path $OUT/regen/pb-think.jsonl     --server-address $SRV || true
$REGEN --reasoning disable --input-file-path $OUT/perfectblend-30k.jsonl --output-file-path $OUT/regen/pb-nothink.jsonl   --server-address $SRV || true

# 5. expand reasoning rows for training
for f in own-think pb-think; do
  [ -s $OUT/regen/$f-exploded.jsonl ] || $PYE $SF/scripts/expand_reasoning_conversations.py \
    --input-file-path $OUT/regen/$f.jsonl --output-file-path $OUT/regen/$f-exploded.jsonl || true
done
wc -l $OUT/regen/*.jsonl | tee /scratch/receipts/regen/counts.txt
echo "REGEN PHASE DONE $(date -u +%FT%TZ)"
