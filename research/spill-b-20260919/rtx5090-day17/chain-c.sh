#!/usr/bin/env bash
# Probe cell C (batched prime arm, day-16 lru shape, texts + [primeseg] receipts) after cell B closes.
D=$HOME/projects/wt-spill-b/research/spill-b-20260919
while [ ! -f $D/rtx5090-day17/gate-b-day16-lru-shape.exit ]; do sleep 10; done
sleep 5
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv > $D/rtx5090-day17/card-before-c.csv
bash $D/run-day17-probe.sh probe-c-batched 1800 --label batched --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150 --cold-turns all --env MEMRA_DEBUG_PRIMESEG=1 --env MEMRA_TTFT_TRACE=1
echo "probe-c exit $(cat $D/rtx5090-day17/probe-c-batched.exit) at $(date -u +%FT%TZ)" >> $D/rtx5090-day17/chain.log
