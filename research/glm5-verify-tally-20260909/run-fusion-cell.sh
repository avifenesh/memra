#!/usr/bin/env bash
set -euo pipefail
cd /root/wt-glm5-verify-tally
R=research/glm5-verify-tally-20260909
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_CUDA_ARCH=100a CARGO_TARGET_DIR=/root/glm5-verify-tally-target MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
while [ ! -f "$R/raw/candidate-build.exit" ]; do sleep 5; done
[ "$(cat "$R/raw/candidate-build.exit")" = 0 ]
cargo fmt --all
git add crates/memra-engine/cu/qmatvec.cu crates/memra-engine/src/glm_spec.rs crates/memra-engine/src/kda.rs crates/memra-engine/src/lib.rs crates/memra-engine/src/bin/glm5_verify_fused6.rs docs/FLAGS.md docs/KERNELS.md "$R/run-fusion-cell.sh"
git -c core.hooksPath=tools/hooks -c user.name=memra-lane -c user.email=lane@localhost commit -m 'perf: gate exact KDA verify six-projection fusion' -m 'Preserve the batched E4M3 dot program, share input quantization and fold macro scales into rounded stores. Candidate defaults off; real-input oracle precedes ABBA timing.' -m 'Claude-Session: https://claude.ai/code/session_01TFyR32RLUiSejCgrPm5nNj'
nice -n 19 cargo build --release --bin memra-server --bin glm5_verify_fused6 -j 16 > "$R/raw/candidate-final-build.log" 2>&1
git rev-parse HEAD > "$R/raw/candidate-source.txt"
sha256sum /root/glm5-verify-tally-target/release/memra-server /root/glm5-verify-tally-target/release/glm5_verify_fused6 > "$R/raw/candidate-binaries.sha256"
flock /tmp/memra-gpu.lock bash "$R/run-fusion-gpu.sh"
