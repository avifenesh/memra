#!/usr/bin/env bash
# Stop the bank-v3 server (ANCHORED absolute-path pattern, this lane's bin dir only)
# and wait for both GPUs to drain to 0 MiB. Ported from toolchain-ab-20260831.
# The anchor is what stops the pattern from self-matching the driving shell.
set -u
PAT="^/home/ubuntu/bankv3/lane/bin/memra-server"
pkill -f "$PAT" 2>/dev/null
for i in $(seq 1 60); do
  pgrep -f "$PAT" >/dev/null || break
  sleep 2
done
pgrep -f "$PAT" >/dev/null && { pkill -9 -f "$PAT"; sleep 3; }
for i in $(seq 1 90); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$U" -le 10 ] && { echo "STOPPED gpus drained (max ${U} MiB)"; exit 0; }
  sleep 2
done
echo "WARN: GPU not fully drained"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
exit 1
