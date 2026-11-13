#!/bin/bash
# dev desk: the decode-shape probe for three builds, sm_120a (the laptop's RTX 5090) or another arch ($1, default
# 120a): k3, the CUTLASS branch head before the fixes (old, 498473abd8) and the fix tree (new), each against its own
# dsv41 kernels, the engine's nvcc flags (dsv41_gpu.cu -fmad=false). Sources staged in $B/src/{k3,old,new} by the
# laptop; CUTLASS from the fix tree's vendored copy.
set -u
ARCH=${1:-120a}
B=$HOME/build/cdec2
NVCC=/usr/local/cuda-13.1/bin/nvcc
GEN="-gencode arch=compute_$ARCH,code=sm_$ARCH -O3 -std=c++17 --expt-relaxed-constexpr"
CI="-I $HOME/build/cfix/crates/memra-engine/third_party/cutlass/include -I $HOME/build/cfix/crates/memra-engine/third_party/cutlass/tools/util/include"
cd $B || exit 1
mkdir -p obj-$ARCH bin-$ARCH log-$ARCH
: > log-$ARCH/rc.txt
ok=1
for b in k3 old new; do
  ( nice -n 10 $NVCC $GEN -fmad=false -c src/$b/dsv41_gpu.cu -o obj-$ARCH/$b-gpu.o > log-$ARCH/$b-gpu.log 2>&1; echo "$b gpu rc $?" >> log-$ARCH/rc.txt ) &
done
for b in old new; do
  ( nice -n 10 $NVCC $GEN -DNDEBUG $CI -c src/$b/dsv41_cutlass.cu -o obj-$ARCH/$b-cutlass.o > log-$ARCH/$b-cutlass.log 2>&1; echo "$b cutlass rc $?" >> log-$ARCH/rc.txt ) &
done
wait
cat log-$ARCH/rc.txt
grep -q "rc [1-9]" log-$ARCH/rc.txt && { grep -h "error" log-$ARCH/*.log | head -30; ok=0; }
$NVCC $GEN -DBUILD_NAME='"k3"' probe.cu obj-$ARCH/k3-gpu.o -o bin-$ARCH/probe-k3 > log-$ARCH/link-k3.log 2>&1 || ok=0
$NVCC $GEN -DCUT probe.cu obj-$ARCH/old-gpu.o obj-$ARCH/old-cutlass.o -o bin-$ARCH/probe-cutlass > log-$ARCH/link-old.log 2>&1 || ok=0
$NVCC $GEN -DCUT -DNEW probe.cu obj-$ARCH/new-gpu.o obj-$ARCH/new-cutlass.o -o bin-$ARCH/probe-new > log-$ARCH/link-new.log 2>&1 || ok=0
grep -h "error" log-$ARCH/link-*.log | head -20
ls -la bin-$ARCH/
sha256sum bin-$ARCH/* | cut -c1-16
echo "PROBE-BUILD-DONE ok=$ok"
