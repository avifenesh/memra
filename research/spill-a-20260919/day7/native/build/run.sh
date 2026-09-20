#!/bin/bash
set -o pipefail
cd /root/wt-spill-a-native-day6
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_CUDA_ARCH=120a
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc
printf "source_commit=b95dc5ee\nfragments=CUDA-TRANSFERS.md\n" > /root/spill-a-day7-receipts/build/source.txt
sha256sum crates/memra-engine/src/tier_transfer.rs crates/memra-engine/src/bin/tier_transfer_gate.rs crates/memra-tier/src/contracts.rs Cargo.lock >> /root/spill-a-day7-receipts/build/source.txt
cargo build -p memra-engine --release --bin tier-transfer-gate -j 16 --offline 2>&1 | tee /root/spill-a-day7-receipts/build/build.log
printf "%s\n" "${PIPESTATUS[0]}" > /root/spill-a-day7-receipts/build/exit
