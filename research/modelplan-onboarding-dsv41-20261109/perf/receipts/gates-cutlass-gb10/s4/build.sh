#!/usr/bin/env bash
# GB10 build lease: the stages' dsv41 binaries for 121a, one after another on one target, from the staged tarballs
# src-sN.tar. usage: build.sh "s1 s2 ..." [NAME]   Done-file out/build[-NAME]/runs.txt, marker BUILD-DONE.
set -u
TAGS=$1
R=$HOME/.cache/dsv41-cfix
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" RUSTUP_TOOLCHAIN=1.97.1
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/build${2:+-$2}; mkdir -p $O; : > $O/runs.txt
for TAG in $TAGS; do
  echo "build $TAG start $(date +%T)" >> $O/runs.txt
  rm -rf $R/src && mkdir -p $R/src $R/bins/$TAG && cd $R/src && tar -xf $R/src-$TAG.tar || { echo "$TAG untar failed" >> $O/runs.txt; continue; }
  find crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs -type f -exec touch {} +
  nice -n 19 timeout 45m cargo build --release --locked -j 12 -p memra-engine --bin dsv41-gate --bin dsv41-bench \
    --bin dsv41-run --bin dsv41-tpbench > $O/build-$TAG.log 2>&1
  rc=$?
  echo "$TAG engine build rc $rc $(date +%T)" >> $O/runs.txt
  if [ $rc -ne 0 ]; then
    grep -E "^error" -A12 $O/build-$TAG.log | head -40 >> $O/runs.txt
  else
    for b in dsv41-gate dsv41-bench dsv41-run dsv41-tpbench; do cp -a $R/target/release/$b $R/bins/$TAG/; done
    sha256sum $R/bins/$TAG/* >> $O/runs.txt
  fi
done
echo "BUILD-DONE $(date +%T)" >> $O/runs.txt
