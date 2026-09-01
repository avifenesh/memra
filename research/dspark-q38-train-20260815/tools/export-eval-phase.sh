#!/usr/bin/env bash
# Phase 3: export latest checkpoint of ARM, normalize for SGLang, serve, gate, eval.
# Usage: export-eval-phase.sh <arm-name> <draft-config> <gpu> [block_size]
# Receipts: /scratch/receipts/eval/<arm>/
set -ux
ARM=$1; CFG=$2; GPU=$3; BLK=${4:-7}
exec >> /home/ubuntu/eval-$ARM.log 2>&1
export PATH=/scratch/venvs/train/bin:$HOME/.local/bin:$PATH
PYE=/scratch/venvs/eval/bin/python
SF=/scratch/repos/SpecForge
R=/scratch/receipts/eval/$ARM
mkdir -p $R
PORT=$((34000+GPU))

# 1. newest checkpoint
CKPT=$(command ls -dt /scratch/ckpt/$ARM/*step* 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "NO CHECKPOINT for $ARM"; exit 1; }
echo "exporting $CKPT"

# 2. export + normalize
cd $SF
specforge export --to hf --checkpoint "$CKPT" --draft-config $CFG \
  --output-dir /scratch/exports/$ARM \
  --embedding-source /scratch/models/qwen38-27b-fp8 \
  --embedding-key model.language_model.embed_tokens.weight
$PYE scripts/gates/normalize_dflash_export.py --config /scratch/exports/$ARM/config.json --block-size $BLK

# 3. serve on the given GPU
tmux kill-session -t eval-srv-$ARM 2>/dev/null || true
tmux new-session -d -s eval-srv-$ARM \
  "CUDA_VISIBLE_DEVICES=$GPU $PYE -m sglang.launch_server --trust-remote-code \
   --model-path /scratch/models/qwen38-27b-fp8 --tp-size 1 \
   --speculative-algorithm DSPARK --speculative-draft-model-path /scratch/exports/$ARM \
   --speculative-dspark-block-size $BLK --speculative-draft-model-quantization unquant \
   --mamba-scheduler-strategy extra_buffer --attention-backend trtllm_mha \
   --reasoning-parser qwen3 --mem-fraction-static 0.85 --port $PORT 2>&1 | tee /home/ubuntu/eval-srv-$ARM.log"
until curl -s --max-time 3 localhost:$PORT/health >/dev/null; do sleep 15; done

# 4. serving gate (one clean block accepted)
$PYE $SF/scripts/gates/run_dflash_chat_serving_gate.py --server-url http://localhost:$PORT --model-path /scratch/models/qwen38-27b-fp8 --served-model q38fp8 --output-path $R/serving-gate.json 2>&1 | tee $R/serving-gate.log || true

# 5. eval: own-sessions + gsm8k anchors (same settings as G1)
$PYE /home/ubuntu/own_bench.py --base-url http://localhost:$PORT --enable-thinking \
  --out $R/own-sessions-t06-think.jsonl 2>&1 | tee $R/own-sessions.log
$PYE -m dflash.benchmark --backend sglang --base-url http://localhost:$PORT \
  --model /scratch/models/qwen38-27b-fp8 --draft-model /scratch/exports/$ARM \
  --dataset gsm8k --max-new-tokens 2048 --temperature 0.6 --top-k 20 --top-p 0.95 \
  --enable-thinking --num-prompts 128 --concurrency 8 --timeout-s 600 2>&1 | tee $R/gsm8k.log

tmux kill-session -t eval-srv-$ARM 2>/dev/null || true
echo "EVAL DONE $ARM $(date -u +%FT%TZ)"
