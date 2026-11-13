#!/usr/bin/env bash
# GB10, one task: the ring gate PR's final tree (TAG) built for 121a and its kernel gate run at once, MEMRA_FULL_PREC=1.
# usage: bgate.sh TAG   Done-file out/bgate-TAG/b.txt, marker BGATE-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-fu1113
D=$HOME/.cache/dsv41-speed/data
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" RUSTUP_TOOLCHAIN=1.97.1
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/bgate-$TAG; mkdir -p $O; : > $O/b.txt
rm -rf $R/src && mkdir -p $R/src $R/bins/$TAG && cd $R/src && tar -xf $R/src-$TAG.tar || { echo "untar failed" >> $O/b.txt; echo BGATE-DONE >> $O/b.txt; exit 1; }
find crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs -type f -exec touch {} +
nice -n 19 timeout 60m cargo build --release --locked -j 16 -p memra-engine --bin dsv41-gate > $O/build.log 2>&1
rc=$?
echo "build rc $rc $(date +%T)" >> $O/b.txt
if [ $rc = 0 ]; then
  cp -a $R/target/release/dsv41-gate $R/bins/$TAG/
  sha256sum $R/bins/$TAG/dsv41-gate >> $O/b.txt
  export MEMRA_FULL_PREC=1
  nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/b.txt 2>&1
  timeout 20m $R/bins/$TAG/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
  echo "kernels rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/kernels.log | tail -1) $(grep -ac '^FAIL' $O/kernels.log) FAIL lines" >> $O/b.txt
  grep -a "^FAIL" $O/kernels.log | head -20 >> $O/b.txt
  grep -a "gemm_ffma_ring\|gemm_ffma2_ring\|gemm_ffma2_partial" $O/kernels.log >> $O/b.txt
else
  grep -E "^error" -A12 $O/build.log | head -40 >> $O/b.txt
fi
echo "BGATE-DONE $(date +%T)" >> $O/b.txt
