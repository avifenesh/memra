#!/usr/bin/env bash
# Assigned tune host only. Own checkout and target; release GO shares the lock.
set -euo pipefail
out=${1:?absolute receipt directory}
: "${CARGO_TARGET_DIR:?lane-owned target required}"
mkdir -p "$out"
if [[ ${2:-} != --locked ]]; then
  exec flock /tmp/memra-gpu.lock nice -n 19 "$0" "$out" --locked
fi
trap 'rc=$?; echo "$rc" > "$out/validation.exit"' EXIT
export MEMRA_CUDA_ARCH=100a MEMRA_GPU_LOCK=/tmp/memra-gpu.lock CARGO_BUILD_JOBS=16
cargo fmt --all
sha256sum crates/memra-engine/src/glm_spec.rs crates/memra-engine/src/glm_spec/prime.rs \
  crates/memra-engine/src/hybrid_forward.rs crates/memra-engine/src/hybrid_forward/glm5_prime.rs \
  crates/memra-engine/src/glm5_tp.rs crates/memra-server/src/worker.rs \
  crates/memra-engine/src/prime_walker.rs crates/memra-server/src/prime_fairness.rs \
  crates/memra-engine/tests/glm5_dflash_session_gpu.rs \
  crates/memra-engine/src/bin/glm5_tp2_box_probe.rs > "$out/source.sha256"
cargo clippy -p memra-engine -p memra-server --lib -j 16 -- -D warnings
cargo test -p memra-engine --lib prime -j 16 -- --test-threads=1
cargo test -p memra-server --lib -j 16 -- --test-threads=1
cargo build -p memra-server -p memra-engine --bin memra-server --bin glm5-tp2-box-probe -j 16
sha256sum "$CARGO_TARGET_DIR/debug/memra-server" \
  "$CARGO_TARGET_DIR/debug/glm5-tp2-box-probe" > "$out/binary.sha256"
# A foreign process still present under the lock is a refusal, never a kill target.
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$out/compute-apps-before.txt"
test ! -s "$out/compute-apps-before.txt"
cargo test -p memra-engine --test glm5_dflash_session_gpu gpu_glm5_prime -j 16 \
  -- --ignored --nocapture --test-threads=1
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$out/compute-apps-after.txt"
test ! -s "$out/compute-apps-after.txt"
cargo fmt --all -- --check
