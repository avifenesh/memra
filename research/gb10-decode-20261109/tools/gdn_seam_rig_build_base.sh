#!/usr/bin/env bash
# Rig: dspark_q38_gate at the stack parent 9390bd3755 (sparse worktree, its own target dir), one
# nvcc build through the shared build lock: the reference for the dspark ladder panic.
set -u
W=$HOME/.cache/memra-gb10-r5/wt-seam-base
S=$HOME/.cache/memra-gb10-r5/seam
export MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$HOME/.cache/memra-gb10-r5/target-seam-base
LOCK=$HOME/.cache/memra-rig-build.lock
cd "$W" || exit 2
echo "$(date -Is) base start" >> "$S/rig-build.status"
flock "$LOCK" nice -n 15 cargo build --release --locked -j 2 -p memra-engine --bin dspark_q38_gate > "$S/bins-base.log" 2>&1
rc=$?
echo "$(date -Is) base rc=$rc" >> "$S/rig-build.status"
if [ "$rc" -eq 0 ]; then
  mkdir -p "$S/bins/9390bd3755-120a"
  cp "$CARGO_TARGET_DIR/release/dspark_q38_gate" "$S/bins/9390bd3755-120a/"
  echo "$(date -Is) base copied" >> "$S/rig-build.status"
fi
echo "$(date -Is) base all done" >> "$S/rig-build.status"
