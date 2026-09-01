#!/usr/bin/env bash
# Poll arm training logs every 5 min; append loss/accept metric lines to receipts CSV.
# Starts before arms exist; exits when block nears end.
set -u
exec >> /home/ubuntu/watch-arms.log 2>&1
R=/scratch/receipts/arms
mkdir -p $R
while true; do
  ts=$(date -u +%FT%TZ)
  for ARM in arm-a-own-cold arm-b-pb-control arm-c-warm36; do
    L=/home/ubuntu/$ARM.log
    [ -f "$L" ] || continue
    # grab the newest metric-looking lines (step/loss/acc patterns from specforge trainer output)
    tail -50 "$L" | grep -aE "step|loss|acc" | tail -3 | sed "s/^/$ts $ARM /" >> $R/metrics.log
    # newest checkpoint marker
    command ls -dt /scratch/ckpt/$ARM/*step* 2>/dev/null | head -1 | sed "s/^/$ts $ARM ckpt /" >> $R/ckpts.log
  done
  sleep 300
done
