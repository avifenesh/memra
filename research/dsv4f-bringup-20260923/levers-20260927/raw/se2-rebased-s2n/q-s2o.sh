#!/usr/bin/env bash
# q-s2o.sh: the DSpark TP/EP gate on the fused lane $1 after its batched-arm engagement fix.
set -u
FSHA=$1
S=/root/rcpt/q-s2o.summary; R=/root/rcpt/moe-fused-dspark-s2o; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-fused-tpep-20260926
git -C /root/lane/t-mft checkout -q -- . ; git -C /root/lane/t-mft checkout -q --detach $FSHA
bash /root/box/build.sh /root/lane/t-mft /root/lane/target-mft mft4
echo "build mft4 $(git -C /root/lane/t-mft rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-mft4.log | tr '\n' ' ')" >> $S
XF=/root/lane/target-mft/release
while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XF/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'arm DB mrow|GATE \[|GATE_DONE|FUSED MOE|MROW' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
echo "S2O_DONE $FSHA" >> $S
