#!/usr/bin/env bash
# GB10 build lease: bs1p = bs1f's kernels compiled with -DV41_SMEM_POISON (the GEMM kernels fill their dynamic shared
# memory with 0xFF before running), dsv41-gate only; the source tree goes back to the unpoisoned file afterwards.
set -u
R=$HOME/.cache/dsv41-bsmma
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/build-poison; mkdir -p $O $R/bins/bs1p; : > $O/runs.txt
E=$R/src/crates/memra-engine
cd $R/src || exit 1
{ echo "#define V41_SMEM_POISON 1"; cat $R/bs1g_dsv41_gpu.cu; } > $E/cu/dsv41_gpu.cu
cp $R/dsv41_gate_e.rs $E/src/bin/dsv41_gate.rs
touch $E/cu/dsv41_gpu.cu
nice -n 19 timeout 20m cargo build --release --locked -j 10 -p memra-engine --bin dsv41-gate > $O/build.log 2>&1
rc=$?
echo "bs1p build rc $rc $(date +%T)" >> $O/runs.txt
[ $rc -eq 0 ] && cp -a $R/target/release/dsv41-gate $R/bins/bs1p/ && sha256sum $R/bins/bs1p/dsv41-gate >> $O/runs.txt
cp $R/bs1g_dsv41_gpu.cu $E/cu/dsv41_gpu.cu
echo "BUILD-POISON-DONE $(date +%T)" >> $O/runs.txt
