#!/usr/bin/env bash
# run2.sh <branch> <commit> <test-target> [filter]: fetch, build one GPU test target, run it.
set -u
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc NVIDIA_TF32_OVERRIDE=0
cd /root/lane/memra && git fetch -q origin $1 && git checkout -q --detach $2
CARGO_TARGET_DIR=/root/lane/target cargo test --release -j 64 -p memra-engine --test $3 --no-run > /root/build-$3.log 2>&1 || { echo BUILD_FAIL; tail -20 /root/build-$3.log; exit 1; }
T=$(grep -o "/root/lane/target/release/deps/$3-[0-9a-f]*" /root/build-$3.log | tail -1)
$T --ignored --nocapture --test-threads=1 ${4:-} 2>&1 | grep -E "EXACT|PASS|panicked|test result|TIMING"
