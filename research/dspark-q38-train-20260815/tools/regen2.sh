#!/usr/bin/env bash
# Recovery: pb prep+regen, own retry at 16K, rebuild train files, relaunch arms (mooncake PATH fixed).
set -ux
exec >> /home/ubuntu/regen2.log 2>&1
export PATH=/scratch/venvs/train/bin:$HOME/.local/bin:$PATH
PYE=/scratch/venvs/eval/bin/python
SF=/scratch/repos/SpecForge
OUT=/scratch/corpus

[ -s $OUT/perfectblend-30k.jsonl ] || { cd $SF && $PYE scripts/prepare_data.py --dataset perfectblend --output-path $OUT/perfectblend-30k.jsonl --sample-size 30000; }
wc -l $OUT/perfectblend-30k.jsonl

for g in 0 1 2 3 4 5 6 7; do
  tmux has-session -t srv$g 2>/dev/null || tmux new-session -d -s srv$g \
    "CUDA_VISIBLE_DEVICES=$g $PYE -m sglang.launch_server --trust-remote-code \
     --model-path /scratch/models/qwen38-27b-fp8 --tp-size 1 \
     --mamba-scheduler-strategy extra_buffer --attention-backend trtllm_mha --reasoning-parser qwen3 \
     --mem-fraction-static 0.85 --port 3000$g 2>&1 | tee /home/ubuntu/srv$g.log"
done
for g in 0 1 2 3 4 5 6 7; do until curl -s --max-time 3 localhost:3000$g/health >/dev/null; do sleep 15; done; done
SRV="localhost:30000 localhost:30001 localhost:30002 localhost:30003 localhost:30004 localhost:30005 localhost:30006 localhost:30007"

REGEN="$PYE $SF/scripts/regenerate_train_data.py --model /scratch/models/qwen38-27b-fp8 --temperature 0 --concurrency 48 --resume"
$REGEN --max-tokens 8192  --reasoning save    --input-file-path $OUT/perfectblend-30k.jsonl --output-file-path $OUT/regen/pb-think.jsonl     --server-address $SRV || true
$REGEN --max-tokens 8192  --reasoning disable --input-file-path $OUT/perfectblend-30k.jsonl --output-file-path $OUT/regen/pb-nothink.jsonl   --server-address $SRV || true
$REGEN --max-tokens 16384 --reasoning save    --input-file-path $OUT/own-prompts.jsonl      --output-file-path $OUT/regen/own-think.jsonl    --server-address $SRV || true
$REGEN --max-tokens 16384 --reasoning save    --input-file-path $OUT/own-prompts-mt.jsonl   --output-file-path $OUT/regen/own-mt-think.jsonl --server-address $SRV || true

for f in own-think pb-think own-mt-think; do
  $PYE $SF/scripts/expand_reasoning_conversations.py --input-file-path $OUT/regen/$f.jsonl --output-file-path $OUT/regen/$f-exploded.jsonl || true
done
wc -l $OUT/regen/*.jsonl | tee /scratch/receipts/regen/counts-final.txt

$PYE /home/ubuntu/build_train_files.py

for g in 0 1 2 3 4 5 6 7; do tmux kill-session -t srv$g 2>/dev/null || true; done
sleep 10
bash /home/ubuntu/launch_arms_inner.sh
echo "REGEN2+ARMS DONE $(date -u +%FT%TZ)"
