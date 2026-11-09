#!/usr/bin/env bash
# Rig: the bins the GDN packed seam removal's GPU cells run (run-spec, kernel-check,
# memra-server), one nvcc build through the shared build lock, then a stable copy for the cells.
# The memra-server suite and the env_audit tests run on the dev desk (devdesk-job.sh).
set -u
W=$HOME/.cache/memra-gb10-r5/wt-seam
S=$HOME/.cache/memra-gb10-r5/seam
export MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$HOME/.cache/memra-gb10-r5/target-seam
LOCK=$HOME/.cache/memra-rig-build.lock
cd "$W" || exit 2
echo "$(date -Is) bins start" >> "$S/rig-build.status"
flock "$LOCK" nice -n 15 cargo build --release --locked -j 2 -p memra-engine --bin run-spec --bin kernel-check -p memra-server --bin memra-server > "$S/bins.log" 2>&1
rc=$?
echo "$(date -Is) bins rc=$rc" >> "$S/rig-build.status"
if [ "$rc" -eq 0 ]; then
  mkdir -p "$S/bins/ed2e0c85ac-120a"
  cp "$CARGO_TARGET_DIR/release/run-spec" "$CARGO_TARGET_DIR/release/kernel-check" \
     "$CARGO_TARGET_DIR/release/memra-server" "$S/bins/ed2e0c85ac-120a/"
  echo "$(date -Is) bins copied" >> "$S/rig-build.status"
fi
echo "$(date -Is) all done" >> "$S/rig-build.status"
