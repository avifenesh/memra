#!/usr/bin/env bash
set -euo pipefail
: "${CARGO_TARGET_DIR:?dedicated target required}"
: "${MEMRA_NVCC:?explicit CUDA toolkit required}"
cuda_root=$(dirname "$(dirname "$MEMRA_NVCC")")
mkdir -p "$CARGO_TARGET_DIR"
helper_sha=$(sha256sum tools/dsv4-launch-boundary.cpp | cut -c1-16)
object=$CARGO_TARGET_DIR/dsv4-launch-boundary-$helper_sha.o
g++ -std=c++17 -O2 -fPIC -I"$cuda_root/include" -c tools/dsv4-launch-boundary.cpp -o "$object"
cargo rustc --release --locked -j2 -p memra-engine --bin dsv4_launch_boundary_gate -- \
    -C "link-arg=$object" -C link-arg=-Wl,--export-dynamic \
    -C link-arg=-Wl,--wrap=memra_dsv4_replay_capture_end \
    -C link-arg=-Wl,--wrap=memra_dsv4_replay_launch \
    -C link-arg=-Wl,--wrap=memra_dsv4_replay_destroy
