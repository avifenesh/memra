#!/usr/bin/env bash
# GB10, one task: the ring gate with padded outputs (TAG's tree, src-TAG.tar), green and red, MEMRA_FULL_PREC=1.
#   green: dsv41-gate built for 121a and `kernels` run on the random fixtures, 4 passes (the first is the receipt)
#   red:   the same gate over the memra#1312 review's mutation build of the kernels (mut-dsv41_gpu.cu: the
#          tree's dsv41_gpu.cu with the ring kernel templated on REVIEW_MUT; 0 is the tree's own kernel text):
#          REVIEW_MUT=0 3 passes, REVIEW_MUT=3 (the chain's column guard dropped: a partial last block stores
#          past N) 20 passes, REVIEW_MUT=1 (the prologue commits a group only for a stage it lands) 3 passes
# usage: padgate.sh TAG   Done-file out/padgate-TAG/p.txt, marker PADGATE-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-fu1113
D=$HOME/.cache/dsv41-speed/data
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" RUSTUP_TOOLCHAIN=1.97.1
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/padgate-$TAG; mkdir -p $O; : > $O/p.txt; : > $O/fails.txt
fin() { echo "PADGATE-DONE $(date +%T)" >> $O/p.txt; exit $1; }
build() {  # build NAME: the tree in $R/src into bins/NAME
  mkdir -p $R/bins/$1
  find crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs -type f -exec touch {} +
  nice -n 19 timeout 40m cargo build --release --locked -j 16 -p memra-engine --bin dsv41-gate > $O/build-$1.log 2>&1
  local rc=$?
  echo "build $1 rc $rc $(date +%T)" >> $O/p.txt
  [ $rc = 0 ] || { grep -E "^error" -A12 $O/build-$1.log | head -40 >> $O/p.txt; fin 1; }
  cp -a $R/target/release/dsv41-gate $R/bins/$1/
  sha256sum $R/bins/$1/dsv41-gate >> $O/p.txt
}
run() {  # run BIN_NAME LABEL PASS [MUT]: one kernels pass; its FAIL entry names go to fails.txt
  local out=$O/kernels-$2-$3
  REVIEW_MUT=${4:-0} timeout 5m $R/bins/$1/dsv41-gate kernels $D/fixtures-gpu/rand --out $out.json > $out.log 2>&1
  local rc=$?
  echo "$2 pass $3 rc $rc $(grep -ao '[0-9]* checks, [0-9]* failed' $out.log | tail -1)" >> $O/p.txt
  grep -a "^FAIL" $out.log | cut -d: -f1 | sed "s|^FAIL |$2 pass $3 |" >> $O/fails.txt
}
rm -rf $R/src && mkdir -p $R/src && cd $R/src && tar -xf $R/src-$TAG.tar || { echo "untar failed" >> $O/p.txt; fin 1; }
build $TAG
export MEMRA_FULL_PREC=1
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/p.txt 2>&1
for p in 1 2 3 4; do run $TAG green $p; done
cp $R/mut-dsv41_gpu.cu crates/memra-engine/cu/dsv41_gpu.cu || { echo "no mutation source" >> $O/p.txt; fin 1; }
build $TAG-mut
for p in 1 2 3; do run $TAG-mut mut0 $p 0; done
for p in $(seq 1 20); do run $TAG-mut mut3 $p 3; done
for p in 1 2 3; do run $TAG-mut mut1 $p 1; done
echo "== entries failing under REVIEW_MUT=3, passes out of 20" >> $O/p.txt
grep "^mut3 " $O/fails.txt | awk '{print $4}' | sort | uniq -c >> $O/p.txt
echo "== entries failing under REVIEW_MUT=1, passes out of 3" >> $O/p.txt
grep "^mut1 " $O/fails.txt | awk '{print $4}' | sort | uniq -c >> $O/p.txt
fin 0
