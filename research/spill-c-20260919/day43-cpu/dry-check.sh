#!/usr/bin/env bash
# Day 43 dry check: one I6G run of the resid shape under the 5090 lock, to prove the plan line prints.
set -uo pipefail
for w in $(seq 1 60); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$apps" = 0 ] && break
  echo "$(date -u +%T) wait $w apps=$apps" ; sleep 60
done
exec 9>/tmp/memra-5090.lock
flock -w 3600 9 || { echo "lock wait timed out"; exit 3; }
echo "lock held $(date -u +%T)"; grep -E 'MemAvailable' /proc/meminfo
env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 /tmp/c40-bins/run-gen-i6 \
  /data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf 55 88 13 --experts-via-tier --expert-bank-stages --expert-bank-host-bytes=8589934592 2>&1 \
  | python3 -c '
import sys, datetime
for line in sys.stdin.buffer:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    sys.stdout.buffer.write(ts.encode() + b"\t" + line); sys.stdout.buffer.flush()
' > /tmp/c40-smoke/i6g.log
echo "rc=${PIPESTATUS[0]} $(date -u +%T)"; grep -E 'MemAvailable' /proc/meminfo
