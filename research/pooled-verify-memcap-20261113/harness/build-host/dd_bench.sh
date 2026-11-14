#!/usr/bin/env bash
# CPU build host, memra#1266: the sm_120a decode-batch-bench (the pooled-vs-serial verify timer) from the fix tree.
set -u
R=<lane>
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target-bench
rm -f $R/bench.rc
( cd $R/src && taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo build --release --locked -j 8 -p memra-engine --bin decode-batch-bench ) > $R/bench.log 2>&1
rc=$?; [ $rc = 0 ] && mkdir -p $R/bins5090/bench && cp -a $R/target-bench/release/decode-batch-bench $R/bins5090/bench/
echo $rc > $R/bench.rc
