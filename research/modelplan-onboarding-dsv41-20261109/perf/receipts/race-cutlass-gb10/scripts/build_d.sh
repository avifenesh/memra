#!/usr/bin/env bash
# Race hunt, GB10 build lease: dsv41-gate (121a) from the staged tarball src-$TAG.tar (TAG from argv or race-tag).
# Done-file out/build-$TAG/runs.txt, marker BUILD-D-DONE.
set -u
R=$HOME/.cache/dsv41-cutlass
TAG=${1:-$(cat $R/race-tag)}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/build-$TAG; mkdir -p $O $R/bins/$TAG; : > $O/runs.txt
echo "build $TAG start $(date +%T)" >> $O/runs.txt
rm -rf $R/src
mkdir -p $R/src && cd $R/src && tar -xf $R/src-$TAG.tar || { echo "untar failed" >> $O/runs.txt; echo "BUILD-D-DONE" >> $O/runs.txt; exit 1; }
find crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs crates/memra-engine/third_party/cutlass/include/cute/arch -type f -exec touch {} +
nice -n 19 timeout 25m cargo build --release --locked -j 12 -p memra-engine --bin dsv41-gate --bin dsv41-bench > $O/build.log 2>&1
rc=$?
echo "engine build rc $rc $(date +%T)" >> $O/runs.txt
if [ $rc -ne 0 ]; then
  grep -E "^error" -A12 $O/build.log | head -60 >> $O/runs.txt
else
  cp -a $R/target/release/dsv41-gate $R/target/release/dsv41-bench $R/bins/$TAG/
  sha256sum $R/bins/$TAG/* >> $O/runs.txt
fi
echo "BUILD-D-DONE $(date +%T)" >> $O/runs.txt
