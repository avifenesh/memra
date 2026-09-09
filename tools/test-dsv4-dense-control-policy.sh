#!/usr/bin/env bash
# CPU-only regression: each actual harness overrides unset/ON before CUDA calls.
set -euo pipefail
: "${CARGO_TARGET_DIR:?dedicated target directory}" "${MEMRA_NVCC:?pinned nvcc}"
out="$CARGO_TARGET_DIR/dsv4-control-policy"
mkdir -p "$out"
for name in dsv4-dense-exact-tail-gate dsv4-dense-tc-gate; do
    "$MEMRA_NVCC" -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off \
        -arch=sm_120a "tools/$name.cu" -lcublasLt -lcublas -ldl -o "$out/$name"
    env -u MEMRA_DSV4_DENSE_FAST MEMRA_DSV4_NORM_FUSE=1 "$out/$name" --check-controls
    env MEMRA_DSV4_DENSE_FAST=1 MEMRA_DSV4_NORM_FUSE=1 "$out/$name" --check-controls
    env MEMRA_DSV4_DENSE_FAST=0 MEMRA_DSV4_NORM_FUSE=0 "$out/$name" --check-controls
    sha256sum "$out/$name"
done
