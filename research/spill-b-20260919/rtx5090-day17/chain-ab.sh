#!/usr/bin/env bash
# Cells A (the gate's default shape) then B (the day-16 lru loop shape: 12 turns, 11,000 + 150 per turn,
# the day-16 cohort), one after the other on the one card.
D=$HOME/projects/wt-spill-b/research/spill-b-20260919
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv > $D/rtx5090-day17/card-before-a.csv
bash $D/run-day17-gate.sh gate-a-default 1500
echo "gate-a exit $(cat $D/rtx5090-day17/gate-a-default.exit) at $(date -u +%FT%TZ)" >> $D/rtx5090-day17/chain.log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv > $D/rtx5090-day17/card-before-b.csv
bash $D/run-day17-gate.sh gate-b-day16-lru-shape 1800 --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150
echo "gate-b exit $(cat $D/rtx5090-day17/gate-b-day16-lru-shape.exit) at $(date -u +%FT%TZ)" >> $D/rtx5090-day17/chain.log
