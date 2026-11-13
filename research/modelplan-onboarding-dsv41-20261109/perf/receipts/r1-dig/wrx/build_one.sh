#!/bin/bash
# wrx (no GPU): build one arm (120a) in src-perf / target-perf from the tree synced into src-TAG and install it as
# perf-bins/TAG (tpbench, gate, bench, run). Run via bin/wrx_build.sh. usage: build_one.sh TAG
set -u
TAG=${1:?tag}
D=/home/access/dsv41
export RUSTUP_HOME=$D/rust/rustup CARGO_HOME=$D/rust/cargo
export PATH=$CARGO_HOME/bin:/usr/local/cuda/bin:/usr/bin:/bin
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$D/target-perf TMPDIR=$D/tmp
rsync -a --delete $D/src-$TAG/ $D/src-perf/
cd $D/src-perf || exit 1
# every source file newer than target-perf's fingerprints: rsync -a keeps the source tree's old mtimes, and the shared
# target dir last built another tree (2026-11-13: a stale build script looked for that tree's cu/dsv41_cutlass.cu)
find crates Cargo.toml Cargo.lock -type f -exec touch {} +
echo "$TAG build start $(date +%H:%M:%S)"
cargo build --release -p memra-engine --bin dsv41-tpbench --bin dsv41-gate --bin dsv41-bench --bin dsv41-run \
  > $D/build-$TAG.log 2>&1
rc=$?
if [ $rc = 0 ]; then
  mkdir -p $D/perf-bins/$TAG
  cp $D/target-perf/release/dsv41-tpbench $D/target-perf/release/dsv41-gate $D/target-perf/release/dsv41-bench \
    $D/target-perf/release/dsv41-run $D/perf-bins/$TAG/
  echo "$TAG gate $(sha256sum $D/perf-bins/$TAG/dsv41-gate | cut -c1-16) tpbench $(sha256sum $D/perf-bins/$TAG/dsv41-tpbench | cut -c1-16)"
fi
grep -E "^error" -A10 $D/build-$TAG.log | head -30
echo "BUILD-DONE $TAG rc $rc $(date +%H:%M:%S)"
