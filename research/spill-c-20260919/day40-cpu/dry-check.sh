#!/usr/bin/env bash
# Day 40 dry check: one ONS run of the attrib shape, under the 5090 lock, to prove the stage lines print.
set -uo pipefail
for w in $(seq 1 20); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$apps" = 0 ] && break
  echo "$(date -u +%T) wait $w apps=$apps" ; sleep 60
done
exec 9>/tmp/memra-5090.lock
flock -w 1800 9 || { echo "lock wait timed out"; exit 3; }
echo "lock held $(date -u +%T)"
env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 /home/avifenesh/projects/wt-spill-c/target/release/run-gen \
  /data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf 55 88 13 --experts-via-tier --expert-bank-stages 2>&1 \
  | python3 -c '
import sys, datetime
for line in sys.stdin.buffer:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    sys.stdout.buffer.write(ts.encode() + b"\t" + line); sys.stdout.buffer.flush()
' > /tmp/c40-smoke/ons.log
echo "rc=${PIPESTATUS[0]} $(date -u +%T)"
