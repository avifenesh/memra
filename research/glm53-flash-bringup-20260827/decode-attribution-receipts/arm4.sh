#!/bin/bash
# Standard decode cell v2: p5 greedy/sampled 192x4 + warm sea continuity + engagement receipts.
TAG=$1; shift
~/cell2.sh $TAG "$@" || exit 1
for i in 0 2 5 7 9; do python3 ~/probe.py warm-$i greedy 24 $i > /dev/null 2>&1; done
python3 ~/steady.py ~/cell-$TAG.log $TAG-p5-greedy  greedy  5 192 4
python3 ~/steady.py ~/cell-$TAG.log $TAG-p5-sampled sampled 5 192 4
python3 ~/steady.py ~/cell-$TAG.log $TAG-p7-greedy  greedy  7 192 4
python3 ~/sea.py $TAG-sea-greedy  greedy  128 4
python3 ~/sea.py $TAG-sea-sampled sampled 128 4
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -g | head -2
echo "loader-law kda warnings: $(grep -c 'loader-law.*kda' ~/cell-$TAG.log)"
echo ARMDONE
