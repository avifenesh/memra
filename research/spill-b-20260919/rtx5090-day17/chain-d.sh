#!/usr/bin/env bash
# Probe cell D: one numeric program on both sides. MEMRA_PREFILL_TICK=8 (documented, authoritative, no floor)
# puts the tick budget below PRIME_MIN_T=16, so prefill_tick's prime branch is never taken and every prompt
# token (the cold 12,350 and the restored suffix alike) rides decode_step (the W1 arm). Cold turn 10 only.
D=$HOME/projects/wt-spill-b/research/spill-b-20260919
while [ ! -f $D/rtx5090-day17/probe-f-gdn-sequential.exit ]; do sleep 10; done
sleep 5
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv > $D/rtx5090-day17/card-before-d.csv
bash $D/run-day17-probe.sh probe-d-tokenwise 3000 --label tokenwise --turns 10 --start-tokens 11000 --grow-tokens 150 --cold-turns 10 --no-cohort --request-timeout-s 900 --env MEMRA_PREFILL_TICK=8 --env MEMRA_DEBUG_PRIMESEG=1 --env MEMRA_TTFT_TRACE=1
echo "probe-d exit $(cat $D/rtx5090-day17/probe-d-tokenwise.exit) at $(date -u +%FT%TZ)" >> $D/rtx5090-day17/chain.log
