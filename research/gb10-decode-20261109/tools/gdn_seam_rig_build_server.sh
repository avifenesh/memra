#!/usr/bin/env bash
# Rig: rebuild the bins the served DFlash verify-graph cell runs (memra-server) at the review-fix
# head, one nvcc build through the shared build lock, then a stable copy for the cells.
set -u
W=$HOME/.cache/memra-gb10-r5/wt-seam
S=$HOME/.cache/memra-gb10-r5/seam
H=0a7d83bf65
export MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$HOME/.cache/memra-gb10-r5/target-seam
LOCK=$HOME/.cache/memra-rig-build.lock
cd "$W" || exit 2
echo "$(date -Is) bins-3 start" >> "$S/rig-build.status"
flock "$LOCK" nice -n 15 cargo build --release --locked -j 2 -p memra-server --bin memra-server > "$S/bins-3.log" 2>&1
rc=$?
echo "$(date -Is) bins-3 rc=$rc" >> "$S/rig-build.status"
if [ "$rc" -eq 0 ]; then
  mkdir -p "$S/bins/$H-120a"
  cp "$CARGO_TARGET_DIR/release/memra-server" "$S/bins/$H-120a/"
  echo "$(date -Is) bins-3 copied" >> "$S/rig-build.status"
fi
echo "$(date -Is) bins-3 all done" >> "$S/rig-build.status"
