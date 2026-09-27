#!/usr/bin/env bash
# q-pair18.sh (WS pod): control for q-pair17's DSpark proposal shas. The same gate on TP/EP from
# the B-row lane before the dense-fast rows kernel (5d21db2a3), on this pod, to tell a pod
# difference from a program change.
while ! grep -q PAIR17_DONE /root/rcpt/q-pair17.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair18.summary; R=/root/rcpt/tp-rows-ctl; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json
git -C /root/lane/memra worktree add -f /root/lane/t-rowsc 5d21db2a3 > /dev/null 2>&1; git -C /root/lane/t-rowsc checkout -q --detach 5d21db2a3
[[ -d /root/lane/target-rowsc ]] || cp -a /root/lane/target-rows /root/lane/target-rowsc
( cd /root/lane/t-rowsc && CARGO_TARGET_DIR=/root/lane/target-rowsc cargo build --release -j 56 -p memra-engine --bin dsv4-gpu-dspark-gate ) > $R/build.log 2>&1
echo "build ctl $(git -C /root/lane/t-rowsc rev-parse --short HEAD) rc=$?" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d /root/lane/target-rowsc/release/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "ctl dspark-tpep $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|FAIL|GATE_DONE' $d/gate.log | head -n 3 | cut -c1-120 | tr '\n' ' ')" >> $S
echo PAIR18_DONE >> $S
