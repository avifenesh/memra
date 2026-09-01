#!/usr/bin/env bash
# G1 remaining cells, spot-proof order (each cell writes receipts immediately; pull-loop streams home).
# Runs ON the box inside tmux. Skips cells whose receipt already exists (resume across spot boxes).
set -ux
R=/scratch/receipts/g1
mkdir -p $R
PY=/scratch/venvs/eval/bin/python
BENCH="$PY -m dflash.benchmark --backend sglang --base-url http://localhost:30000 \
  --model /scratch/models/qwen38-27b-fp8 --draft-model /scratch/models/radixark-q38-dspark \
  --max-new-tokens 2048 --num-prompts 128 --concurrency 8 --timeout-s 600"

until curl -s --max-time 3 localhost:30000/health >/dev/null; do sleep 10; done

# 1. greedy arm, spec ON (thinking on, temp 0)
[ -s $R/gsm8k-t0-think.log ] || $BENCH --dataset gsm8k --temperature 0 --enable-thinking 2>&1 | tee $R/gsm8k-t0-think.log
[ -s $R/own-sessions-t0-think.jsonl ] || $PY /home/ubuntu/own_bench.py --enable-thinking --temperature 0 \
  --out $R/own-sessions-t0-think.jsonl 2>&1 | tee $R/own-sessions-t0-think.log
# 2. own-sessions nothink arm (serving also runs nothink)
[ -s $R/own-sessions-t06-nothink.jsonl ] || $PY /home/ubuntu/own_bench.py --temperature 0.6 \
  --out $R/own-sessions-t06-nothink.jsonl 2>&1 | tee $R/own-sessions-t06-nothink.log
# 3. spec-ON repeat for variance (gsm8k)
[ -s $R/gsm8k-t06-think-r2.log ] || $BENCH --dataset gsm8k --temperature 0.6 --top-k 20 --top-p 0.95 \
  --enable-thinking 2>&1 | tee $R/gsm8k-t06-think-r2.log

# 4. spec-OFF baselines: restart server plain, same workloads (denominators)
tmux kill-session -t serve 2>/dev/null || true
sleep 5
tmux new-session -d -s serve "env CUDA_VISIBLE_DEVICES=0 $PY -m sglang.launch_server \
  --trust-remote-code --model-path /scratch/models/qwen38-27b-fp8 --tp-size 1 \
  --mamba-scheduler-strategy extra_buffer --attention-backend flashinfer \
  --mem-fraction-static 0.85 --port 30000 2>&1 | tee /home/ubuntu/serve-plain.log"
until curl -s --max-time 3 localhost:30000/health >/dev/null; do sleep 10; done
[ -s $R/gsm8k-t06-think-specoff.log ] || $BENCH --dataset gsm8k --temperature 0.6 --top-k 20 --top-p 0.95 \
  --enable-thinking 2>&1 | tee $R/gsm8k-t06-think-specoff.log
[ -s $R/own-sessions-t06-think-specoff.jsonl ] || $PY /home/ubuntu/own_bench.py --enable-thinking \
  --temperature 0.6 --out $R/own-sessions-t06-think-specoff.jsonl 2>&1 | tee $R/own-sessions-t06-think-specoff.log

echo "BENCH-QUEUE-DONE $(date -u +%FT%TZ)" | tee $R/QUEUE-DONE
