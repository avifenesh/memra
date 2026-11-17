#!/bin/bash
# dev desk: the memra#1324 probe, sm_120a, the engine's nvcc flags for dsv41_gpu.cu (-O3 -fmad=false).
# Sources staged in ~/.cache/dsv41-moe1324/probe/src/{main,base,head}/dsv41_gpu.cu by the laptop.
# usage: build.sh [builds...]   (default: main base)
set -u
P=$HOME/.cache/dsv41-moe1324/probe
NVCC=/usr/local/cuda-13.1/bin/nvcc
GEN="-gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr -fmad=false"
cd $P || exit 1
mkdir -p obj bin log
B="${*:-main base}"
: > log/rc.txt
for b in $B; do
  ( taskset -c 14-31 nice -n 10 $NVCC $GEN -c src/$b/dsv41_gpu.cu -o obj/$b-gpu.o > log/$b-gpu.log 2>&1; echo "$b gpu rc $?" >> log/rc.txt ) &
done
wait
ok=1
grep -q "rc [1-9]" log/rc.txt && { grep -h "error" log/*-gpu.log | head -30; ok=0; }
for b in $B; do
  def=""; [ $b = base ] && def="-DBASE"; [ $b = head ] && def="-DHEAD"
  taskset -c 14-31 nice -n 10 $NVCC $GEN $def probe1324.cu obj/$b-gpu.o -o bin/probe-$b > log/link-$b.log 2>&1 || { ok=0; grep -h error log/link-$b.log | head; }
done
cat log/rc.txt
ls -la bin/
sha256sum bin/* | cut -c1-16
echo "PROBE-BUILD-DONE ok=$ok"
