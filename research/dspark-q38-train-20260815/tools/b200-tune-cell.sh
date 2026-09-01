#!/usr/bin/env bash
# One tuning cell: restart memra server with env overrides, warmup + N timed 800-tok generations.
# Usage: b200-tune-cell.sh <cell-name> [ENV=VAL ...]   Logs JSONL to /scratch/receipts/memra-b200-tune/cells.jsonl
set -u
NAME=$1; shift
ENVS="$*"
R=/scratch/receipts/memra-b200-tune
mkdir -p $R
tmux kill-session -t memrast 2>/dev/null
sleep 4
tmux new-session -d -s memrast "cd /scratch/repos/memra && CUDA_VISIBLE_DEVICES=6 MEMRA_MODELS=q38fp8=/scratch/models/qwen38-27b-fp8 $ENVS ./target/release/memra-server 2>&1 | tee /home/ubuntu/memra-st.log"
for i in $(seq 1 60); do curl -s --max-time 3 localhost:8080/v1/models >/dev/null 2>&1 && break; sleep 10; done
BODY='{"model":"q38fp8","messages":[{"role":"user","content":"Explain how a CPU branch predictor works, in detail."}],"max_tokens":800,"temperature":0}'
curl -s --max-time 300 localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "$BODY" >/dev/null  # warmup
for rep in 1 2 3; do
  OUT=$(curl -s --max-time 300 localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "$BODY")
  echo "$OUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
u = d['usage']
row = {'cell': '$NAME', 'rep': $rep, 'envs': '$ENVS',
       'completion_tokens': u['completion_tokens'], 'elapsed_s': round(u['elapsed_s'], 3),
       'tok_s': round(u['completion_tokens'] / u['elapsed_s'], 2), 'spec': u.get('spec')}
print(json.dumps(row))
" >> $R/cells.jsonl
done
tail -3 $R/cells.jsonl
