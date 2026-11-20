#!/usr/bin/env bash
# b23d: build the kernel bench against the perf source's .cu files (same flags as build.rs)
set -e
cd $HOME/hy4flash/perf/kbench
export PATH=/usr/local/cuda-13.0/bin:$PATH
CU=$HOME/hy4flash/perf/src/crates/memra-engine/cu
F="-gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr"
nvcc $F -c $CU/hy4f_route.cu -o route.o &
nvcc $F -fmad=false -c $CU/hy4f_gpu.cu -o gpu.o &
nvcc $F -c kbench.cu -o kbench.o &
wait
nvcc $F -o kbench kbench.o route.o gpu.o
echo KBENCH_BUILT
