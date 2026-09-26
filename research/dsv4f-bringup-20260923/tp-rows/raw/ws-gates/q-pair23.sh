#!/usr/bin/env bash
# q-pair23.sh (WS pod): memra #710 lane 7150cb5f1, the PP-2 rows gate (the hoisted compressor
# projections run in the PP B-row attention too) and the DSpark gate on PP-2.
while ! grep -q PAIR22_DONE /root/rcpt/q-pair22.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair23.summary; R=/root/rcpt/rows-pp-7150cb5f1; mkdir -p $R
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; X=/root/lane/target-rg/release
echo "binary $(git -C /root/lane/t-rg rev-parse --short HEAD) gate $(sha256sum $X/dsv4_rows_gate | cut -c1-16)" >> $S
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3000 $X/dsv4_rows_gate $M /root/box/tape-rebuild.txt 24 64 9>&- > $R/gate.log 2>&1 )
echo "rows-pp rc=$? $(grep -hE 'PASS|FAILED|DIVERGENCE|TIME rep=1|panicked' $R/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
d=$R/dspark-pp; mkdir -p $d; while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device; /root/box/gate.sh $d $X/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served )
echo "dspark-pp $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|FAIL|GATE_DONE' $d/gate.log | head -n 3 | cut -c1-120 | tr '\n' ' ')" >> $S
echo PAIR23_DONE >> $S
