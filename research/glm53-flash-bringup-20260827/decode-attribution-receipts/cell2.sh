#!/bin/bash
# usage: cell2.sh <tag> [extra env assignments...]
# Restarts memra-server WARM (never drops caches) with the base serving env + extras.
TAG=$1; shift
pkill -x memra-server; sleep 5; pkill -9 -x memra-server 2>/dev/null; sleep 3
cd ~/memra
LOG=~/cell-$TAG.log
: > $LOG
env "$@" MEMRA_SPILL_STATS=1 CUDA_VISIBLE_DEVICES=0,1 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18400 \
  MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 \
  setsid nohup ./target/release/memra-server > $LOG 2>&1 < /dev/null &
disown
for i in $(seq 1 600); do grep -q 'listening on' $LOG && break; grep -qE '^\[server\] .*(error|failed)|panicked' $LOG && break; sleep 2; done
if ! grep -q 'listening on' $LOG; then echo "LOAD FAILED after $((i*2))s"; tail -20 $LOG; exit 1; fi
echo "LOAD $TAG: ready after ~$((i*2))s"
grep -E '\[moe\] resident|moe-cache\] size-aware|decode wave cap|EAGER-ONLY|spill-pread|PP |pp ' $LOG | head -10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
