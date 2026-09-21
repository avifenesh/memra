#!/usr/bin/env bash
# Probe cells E (restore points on the 12,350 prompt: on-grid 12288/12320, off-grid 12200/12250/12300) and
# F (the day-16 lru chain under MEMRA_GDN_CHUNKED=0 on both boots, the split-invariant sequential scan).
D=$HOME/projects/wt-spill-b/research/spill-b-20260919
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv > $D/rtx5090-day17/card-before-e.csv
bash $D/run-day17-probe.sh probe-e-restore-points 1500 --label restore-points --turns 10 --start-tokens 11000 --grow-tokens 150 --cold-turns 10 --no-chain --no-cohort --restore-points 12288,12320,12200,12250,12300 --env MEMRA_DEBUG_PRIMESEG=1 --env MEMRA_TTFT_TRACE=1
echo "probe-e exit $(cat $D/rtx5090-day17/probe-e-restore-points.exit) at $(date -u +%FT%TZ)" >> $D/rtx5090-day17/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv > $D/rtx5090-day17/card-before-f.csv
bash $D/run-day17-probe.sh probe-f-gdn-sequential 1800 --label gdn-sequential --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150 --cold-turns all --env MEMRA_GDN_CHUNKED=0 --env MEMRA_DEBUG_PRIMESEG=1 --env MEMRA_TTFT_TRACE=1
echo "probe-f exit $(cat $D/rtx5090-day17/probe-f-gdn-sequential.exit) at $(date -u +%FT%TZ)" >> $D/rtx5090-day17/chain.log
