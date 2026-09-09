#!/usr/bin/env bash
# CPU-only regression: each actual harness overrides unset/ON before CUDA calls.
set -euo pipefail
: "${CARGO_TARGET_DIR:?dedicated target directory}" "${MEMRA_NVCC:?pinned nvcc}"
out="$CARGO_TARGET_DIR/dsv4-control-policy"
mkdir -p "$out"
cuda_root=$(dirname "$(dirname "$(readlink -f "$MEMRA_NVCC")")")
# Driver-based harnesses still take the CPU-only early exit. NVIDIA's link stub
# lets GPU-free hosted CI load them; no driver API is called by --check-controls.
ln -sf "$cuda_root/lib64/stubs/libcuda.so" "$out/libcuda.so.1"
export LD_LIBRARY_PATH="$out:${LD_LIBRARY_PATH:-}"
for name in dsv4-dense-exact-tail-gate dsv4-dense-tc-gate dsv4-dense-tc-gate-r4 dsv4-dense-tc-gate-r7-driver dsv4-dense-tc-gate-r8-driver dsv4-dense-tc-gate-r9-driver; do
    "$MEMRA_NVCC" -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off \
        -arch=sm_120a "tools/$name.cu" -lcublasLt -lcublas -ldl -L"$cuda_root/lib64/stubs" -lcuda -o "$out/$name"
    env -u MEMRA_DSV4_HC_DOT_SPLIT -u MEMRA_DSV4_DENSE_FAST MEMRA_DSV4_NORM_FUSE=1 "$out/$name" --check-controls
    env MEMRA_DSV4_HC_DOT_SPLIT=16 MEMRA_DSV4_DENSE_FAST=1 MEMRA_DSV4_NORM_FUSE=1 "$out/$name" --check-controls
    env MEMRA_DSV4_HC_DOT_SPLIT=0 MEMRA_DSV4_DENSE_FAST=0 MEMRA_DSV4_NORM_FUSE=0 "$out/$name" --check-controls
    sha256sum "$out/$name"
done
