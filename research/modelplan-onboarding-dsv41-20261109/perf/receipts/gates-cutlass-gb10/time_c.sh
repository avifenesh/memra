#!/usr/bin/env bash
# GB10 TIMING PROXY (one lease, <= 30 min): arm A (ARM_A: k3 | bs1) against B = this branch's build, A B B A:
#   dsv41-bench gemm (the FP8/FP4 GEMMs alone at the trunk's shapes) and dsv41-bench layer (layer 2's 4,096-token
#   prefill block REPS times + 32 one-token decode steps), every run hashed, clocks every 250 ms.
set -u
TAG=$(cat $HOME/.cache/dsv41-cutlass/build-tag)
ARM=$(cat $HOME/.cache/dsv41-cutlass/time-arm)
R=$HOME/.cache/dsv41-cutlass
S=$HOME/.cache/dsv41-speed
D=$S/data; [ -d $D ] || D=$HOME/.cache/dsv41-cutlass/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
REPS=${REPS:-8}
if [ $ARM = k3 ]; then AD=$S/bins/k3; else AD=$HOME/.cache/dsv41-bsmma/bins/bs1f; fi
[ -d $AD ] || AD=$R/arms/$ARM
O=$R/out/time-$ARM-$TAG; mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1
echo "A $AD $(sha256sum $AD/dsv41-bench | cut -c1-16)" > $O/arms.txt
echo "B $R/bins/$TAG $(sha256sum $R/bins/$TAG/dsv41-bench | cut -c1-16)" >> $O/arms.txt
i=0
for a in A B B A; do
  i=$((i+1))
  if [ $a = A ]; then BD=$AD; else BD=$R/bins/$TAG; fi
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv,noheader -lms 250 > $O/run$i-$a.smi 2>&1 &
  SP=$!
  timeout 4m $BD/dsv41-bench gemm --reps 20 --out $O/run$i-$a-gemm.json > $O/run$i-$a-gemm.log 2>&1
  echo "run$i-$a gemm rc $? $(date +%T)" >> $O/runs.txt
  timeout 5m $BD/dsv41-bench layer $C --ids $R/data/prefill-4096.i32 --token-map $TM --layers 2 \
    --prefix 4096 --reps $REPS --decode 32 --out $O/run$i-$a-layer.json > $O/run$i-$a-layer.log 2>&1
  echo "run$i-$a layer rc $? $(date +%T)" >> $O/runs.txt
  kill $SP
done
echo "TIME-C-DONE $(date +%T)" >> $O/runs.txt
