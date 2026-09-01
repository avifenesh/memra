#!/usr/bin/env bash
# Auto-chain: wait for corpus gen + hidden capture to finish, then train on GPU0.
# Capture only prints CAPTURE DONE after the gen driver's DONE line, and the gen
# wrapper kills the server on exit — so GPU0 is free and the lock is droppable
# by the time this fires.
set -euo pipefail
cd "$HOME/models/ornith15"

while ! grep -q "CAPTURE DONE" mtp-train/capture.log 2>/dev/null; do sleep 120; done
sleep 30

exec 9>/tmp/memra-gpu.lock
flock -w 600 9 || { echo "FATAL: gpu lock busy after 10m"; exit 1; }

RC=0
env CUDA_VISIBLE_DEVICES=0 python3 mtp-train/train_mtp.py \
  --bf16-dir bf16 \
  --hiddens-dir mtp-train/hiddens \
  --corpus mtp-train/corpus.jsonl \
  --out-dir mtp-train/train-out > mtp-train/train.log 2>&1 || RC=$?
echo "CHAIN DONE rc=$RC"
exit "$RC"
