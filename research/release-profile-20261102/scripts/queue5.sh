#!/usr/bin/env bash
# Tokenizer-focused A/B: 60 timed /v1/tokenize calls per boot with server CPU time, ABBAAB.
set -u
cd ~/.cache/memra-relprofile
i=0
for arm in A B B A A B; do
  i=$((i+1))
  flock ~/.cache/memra-rig-build.lock python3 host_bench.py $arm $i tok > logs/hosttok-$arm-$i.out 2>&1 || echo "round $i arm $arm failed rc=$?"
done
echo queue5 done
