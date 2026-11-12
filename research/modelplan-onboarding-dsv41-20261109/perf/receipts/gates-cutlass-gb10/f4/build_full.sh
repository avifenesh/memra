#!/usr/bin/env bash
# GB10 build lease: the cutlass branch's dsv41 binaries for 121a (+ the memra CLI for CheckpointParity) from the
# staged tarball src-$TAG.tar. Target seeded once from the bsmma lane's (deps). Done-file out/build-$TAG/runs.txt,
# marker BUILD-F-DONE.
set -u
TAG=${1:-f4}
R=$HOME/.cache/dsv41-cutlass
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/build-$TAG; mkdir -p $O $R/bins/$TAG; : > $O/runs.txt
echo "build $TAG start $(date +%T)" >> $O/runs.txt
[ -d $R/target ] || cp -a $HOME/.cache/dsv41-bsmma/target $R/target
rm -rf $HOME/.cache/dsv41-cutlass/src
mkdir -p $R/src && cd $R/src && tar -xf $R/src-$TAG.tar || { echo "untar failed" >> $O/runs.txt; echo "BUILD-F-DONE" >> $O/runs.txt; exit 1; }
# the tarball keeps the laptop's mtimes; the engine's sources are made newer than any fingerprint
find crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs -type f -exec touch {} +
nice -n 19 timeout 25m cargo build --release --locked -j 12 -p memra-engine --bin dsv41-gate --bin dsv41-bench \
  --bin dsv41-run --bin dsv41-tpbench > $O/build.log 2>&1
rc=$?
echo "engine build rc $rc $(date +%T)" >> $O/runs.txt
if [ $rc -ne 0 ]; then
  grep -E "^error" -A12 $O/build.log | head -60 >> $O/runs.txt
else
  for b in dsv41-gate dsv41-bench dsv41-run dsv41-tpbench; do cp -a $R/target/release/$b $R/bins/$TAG/; done
  nice -n 19 timeout 10m cargo build --release --locked -j 12 -p memra-cli --bin memra > $O/build-cli.log 2>&1
  echo "memra-cli build rc $? $(date +%T)" >> $O/runs.txt
  [ -x $R/target/release/memra ] && cp -a $R/target/release/memra $R/bins/$TAG/
  sha256sum $R/bins/$TAG/* >> $O/runs.txt
fi
echo "BUILD-F-DONE $(date +%T)" >> $O/runs.txt
