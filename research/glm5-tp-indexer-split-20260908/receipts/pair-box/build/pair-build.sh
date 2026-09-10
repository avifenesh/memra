#!/bin/bash
set -euo pipefail
cd /root/wt-glm5-indexer-split
export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MEMRA_CUDA_ARCH=100a MEMRA_GPU_LOCK=/tmp/memra-gpu.lock MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc
mkdir -p .lane
sha256sum -c .lane/pair-source.sha256 > .lane/source-check.log
nice -n 19 cargo build --release -p memra-engine --bin glm5-tp2-box-probe -j 16 > .lane/pair-build.log 2>&1
nice -n 19 cargo fmt --all -- --check > .lane/pair-fmt.log 2>&1
nice -n 19 cargo test --release -p memra-engine --test glm5_tp_indexer_split -j 16 > .lane/pair-cpu.log 2>&1
sha256sum target/release/glm5-tp2-box-probe > .lane/pair-binary.sha256
