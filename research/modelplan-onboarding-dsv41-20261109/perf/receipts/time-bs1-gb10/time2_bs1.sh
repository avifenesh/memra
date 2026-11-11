#!/usr/bin/env bash
# GB10 (one lease, <= 30 min): TIMING PROXY for bs1 against k3 (A = the speed lane's k3 binaries, B = bs1), A B B A:
#   1. dsv41-bench layer: the trunk's prefill of the 4,096-token prompt layer by layer (each layer loaded and freed,
#      as dsv41-run does: the resident trunk does not fit GB10 beside the page cache), layer 2's block rerun REPS
#      times on the same input, then 32 one-token decode steps; every run hashed
#   2. dsv41-bench gemm: the block-scaled GEMMs alone at the trunk's shapes (dense FP8 at M 1 / 64 / 4096 / 8192,
#      the experts at decode and at a 4k and a 32k prefill's mean rows per expert), with sha256s
# Clocks / power / temperature sampled every 250 ms. A card under its power cap steps its clock down after 0.15-1.5 s
# of load: the reps after the first two are the steady figures.
set -u
R=$HOME/.cache/dsv41-bsmma
S=$HOME/.cache/dsv41-speed
D=$S/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
REPS=${REPS:-8}
O=$R/out/time2-bs1; mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1
echo "A $S/bins/k3 $(sha256sum $S/bins/k3/dsv41-bench | cut -c1-16)" > $O/arms.txt
echo "B $R/bins/bs1 $(sha256sum $R/bins/bs1/dsv41-bench | cut -c1-16)" >> $O/arms.txt
i=0
for a in A B B A; do
  i=$((i+1))
  if [ $a = A ]; then BD=$S/bins/k3; else BD=$R/bins/bs1; fi
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv,noheader -lms 250 > $O/run$i-$a.smi 2>&1 &
  SP=$!
  timeout 4m $BD/dsv41-bench gemm --reps 20 --out $O/run$i-$a-gemm.json > $O/run$i-$a-gemm.log 2>&1
  echo "run$i-$a gemm rc $? $(date +%T)" >> $O/runs.txt
  timeout 5m $BD/dsv41-bench layer $C --ids $R/prefill-4096.i32 --token-map $TM --layers 2 --prefix 4096 --reps $REPS \
    --decode 32 --out $O/run$i-$a-layer.json > $O/run$i-$a-layer.log 2>&1
  echo "run$i-$a layer rc $? $(date +%T)" >> $O/runs.txt
  kill $SP
done
echo "TIME2-BS1-DONE $(date +%T)" >> $O/runs.txt
