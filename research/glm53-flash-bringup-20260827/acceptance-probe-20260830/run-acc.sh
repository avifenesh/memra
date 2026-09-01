#!/bin/bash
# card3-lane CELL 2 runner: engine-level acceptance probe on the real artifact.
# Usage: run-acc.sh <tag> <prompts_dir> [MEMRA_FRSPEC_TRIM=<ranks.txt> ...extra env]
# Count-based only. CARD3_HOLD_FILE makes the probe pause between runs while the
# co-tenant lane's timing marker exists.
set -u
TAG=$1; PDIR=$2; shift 2
LANE=/root/card3-lane
OUT=$LANE/out/acc-$TAG
LOG=$LANE/logs/probe-$TAG.log
mkdir -p "$OUT"
env "$@" \
  MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1 \
  MEMRA_ST_PINNED=1 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 \
  CUDA_VISIBLE_DEVICES=3 NVIDIA_TF32_OVERRIDE=0 \
  CARD3_HOLD_FILE=/root/TIMING-IN-FLIGHT CARD3_MAX_NEW=${CARD3_MAX_NEW:-128} \
  CARD3_KS=${CARD3_KS:-1,2,3,4,5,6,7} \
  nohup /root/memra-card3/target/release/glm5-card3-probe \
    /root/models/glm53-nvfp4 "$PDIR" "$OUT" > "$LOG" 2>&1 &
PID=$!
disown
echo "$PID" > $LANE/probe-$TAG.pid
echo "probe $TAG pid=$PID log=$LOG out=$OUT"
