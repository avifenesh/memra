#!/bin/bash
# 5090 (under gpulock): the fp32 FMA family microbench, kpc / r1 / r2, with the card's clocks around it; from b3 on
# also r2's kernel gate (dev-desk 120a build) on the fixtures.
# usage: run_bench.sh TAG
set -u
D=/home/avifenesh/.cache/dsv41-memra/r1-dig/bench
T=${1:-b1}
cd $D || exit 1
{
  date +%FT%T
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv
  ./ffma_bench 21
  echo "rc $?"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv
  date +%FT%T
} > $D/$T.txt 2>&1
if [ "$T" != b1 ] && [ "$T" != b2 ] && [ -x /home/avifenesh/.cache/dsv41-memra/r1-dig/bins-r2-120a/dsv41-gate ]; then
  G=/home/avifenesh/.cache/dsv41-memra/r1-dig/kgate-5090-$T
  mkdir -p $G
  sha256sum /home/avifenesh/.cache/dsv41-memra/r1-dig/bins-r2-120a/dsv41-gate > $G/bin.sha256
  MEMRA_FULL_PREC=1 /home/avifenesh/.cache/dsv41-memra/r1-dig/bins-r2-120a/dsv41-gate kernels \
    /home/avifenesh/.cache/dsv41-memra/r1-dig/fixtures/rand --out $G/kernels.json > $G/kernels.log 2>&1
  echo "kernel gate rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $G/kernels.log | tail -1)" >> $D/$T.txt
fi
# the engine's own timing of the fp32 FMA calls (dsv41-bench gemm, router rows 1/4/5/8 with output hashes), by arm
for arm in kpc r1 r2 r2 r1 kpc; do
  B=/home/avifenesh/.cache/dsv41-memra/r1-dig/bins-$arm-120a/dsv41-bench
  [ -x $B ] || continue
  G=/home/avifenesh/.cache/dsv41-memra/r1-dig/gemm-5090-$T
  mkdir -p $G
  n=$(ls $G | grep -c "^$arm-")
  $B gemm --reps 30 --out $G/$arm-$n.json > $G/$arm-$n.log 2>&1
  echo "gemm $arm run $n rc $?" >> $D/$T.txt
  grep -a "gemm ffma" $G/$arm-$n.log >> $D/$T.txt
done
echo "BENCH-DONE $T" >> $D/$T.txt
