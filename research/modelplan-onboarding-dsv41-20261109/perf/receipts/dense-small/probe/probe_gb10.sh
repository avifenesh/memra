#!/usr/bin/env bash
# GB10: build the decode-shape probe for 121a from the staged sources (probe-src.tgz: src/{k3,old,new}, probe.cu,
# cutlass-inc.tgz), then the new build's byte check and the dense and dense-set cells of the three builds interleaved
# (k3 old new / new old k3, twice). Done-file out/probe-$TAG/runs.txt, marker PROBE-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-cfix
P=$R/probe-$TAG
O=$R/out/probe-$TAG; mkdir -p $O; : > $O/runs.txt
NVCC=/usr/local/cuda/bin/nvcc
GEN="-gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr"
rm -rf $P; mkdir -p $P && cd $P && tar -xzf $R/probe-src.tgz && tar -xzf cutlass-inc.tgz
CI="-I $P/cutlass/include -I $P/cutlass/tools/util/include"
mkdir -p obj bin log
for b in k3 old new; do
  ( nice -n 10 $NVCC $GEN -fmad=false -c src/$b/dsv41_gpu.cu -o obj/$b-gpu.o > log/$b-gpu.log 2>&1; echo "$b gpu rc $?" >> $O/runs.txt ) &
done
for b in old new; do
  ( nice -n 10 $NVCC $GEN -DNDEBUG $CI -c src/$b/dsv41_cutlass.cu -o obj/$b-cutlass.o > log/$b-cutlass.log 2>&1; echo "$b cutlass rc $?" >> $O/runs.txt ) &
done
wait
$NVCC $GEN -DBUILD_NAME='"k3"' probe.cu obj/k3-gpu.o -o bin/probe-k3 > log/link-k3.log 2>&1; echo "link k3 rc $?" >> $O/runs.txt
$NVCC $GEN -DCUT probe.cu obj/old-gpu.o obj/old-cutlass.o -o bin/probe-cutlass > log/link-old.log 2>&1; echo "link old rc $?" >> $O/runs.txt
$NVCC $GEN -DCUT -DNEW probe.cu obj/new-gpu.o obj/new-cutlass.o -o bin/probe-new > log/link-new.log 2>&1; echo "link new rc $?" >> $O/runs.txt
sha256sum bin/* >> $O/runs.txt
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/runs.txt 2>&1
timeout 900 bin/probe-new check > $O/check.txt 2>&1; echo "check rc $? $(grep '# check' $O/check.txt)" >> $O/runs.txt
i=0
for order in "k3 cutlass new" "new cutlass k3" "k3 cutlass new" "new cutlass k3"; do
  for b in $order; do
    i=$((i+1))
    timeout 900 bin/probe-$b ${2:-all} 64 > $O/run$i-$b.txt 2>&1; echo "run$i $b rc $? $(date +%T)" >> $O/runs.txt
  done
done
echo "PROBE-DONE $(date +%T)" >> $O/runs.txt
