#!/usr/bin/env bash
# Stop the ab-lane server (bracketed-basename pattern, this lane's path only)
# and wait for both GPUs to drain to 0 MiB.
set -u
pkill -f "^/home/ubuntu/toolchain-ab/bin/memra-server" 2>/dev/null
for i in $(seq 1 60); do
  pgrep -f "^/home/ubuntu/toolchain-ab/bin/memra-server" >/dev/null || break
  sleep 2
done
pgrep -f "^/home/ubuntu/toolchain-ab/bin/memra-server" >/dev/null && { pkill -9 -f "^/home/ubuntu/toolchain-ab/bin/memra-server"; sleep 3; }
for i in $(seq 1 60); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$U" -le 10 ] && { echo "STOPPED gpus drained (max ${U} MiB)"; exit 0; }
  sleep 2
done
echo "WARN: GPU not fully drained"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
exit 1
