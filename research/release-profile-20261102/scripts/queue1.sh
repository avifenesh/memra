#!/usr/bin/env bash
# Build A/B queue: B cold 1, then interleaved ABBAAB touch rebuilds.
set -u
cd ~/.cache/memra-relprofile
waitfor --gone 656404 --timeout 3h --quiet
python3 measure_build.py B cold 1 > /dev/null 2>&1
i=0
for arm in A B B A A B; do
  i=$((i+1)); python3 measure_build.py $arm touch-server $i > /dev/null 2>&1
done
i=0
for arm in A B B A A B; do
  i=$((i+1)); python3 measure_build.py $arm touch-engine $i > /dev/null 2>&1
done
echo queue1 done
