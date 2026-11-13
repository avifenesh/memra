#!/usr/bin/env bash
# GB10 (b23d, under one sparklock lease): the r1 bisect's research binary r1x = r1 (5a4e78bb32) + MEMRA_R1X_SEG, the call
# sites that take the K-segment order (router, comp, head; unset = all three = r1). Build for 121a, then the layer
# gate's saved tensors: "none" must be k3's set byte for byte (the switch reaches every site), then the router-only
# and comp-only arms against k3's and r1's sets (which site moves which tensor). Out: out/r1x/.
set -u
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/r1x
mkdir -p $O $R/bins/r1x
: > $O/build.txt
T0=$(date +%s)
cd $R/src || exit 1
tar -x -f "$R/r1x-overlay.tar"
touch crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/dsv41_gpu.rs crates/memra-engine/src/dsv41_gpu/tp.rs \
  crates/memra-engine/src/dsv41_gpu/decode.rs crates/memra-engine/src/bin/dsv41_gate.rs \
  crates/memra-engine/src/bin/dsv41_run.rs crates/memra-engine/src/bin/dsv41_tpbench.rs
echo "build start $(date +%T)" >> $O/build.txt
if ! nice -n 19 cargo build --release --locked -j 10 -p memra-engine --bin dsv41-gate --bin dsv41-run > $O/build.log 2>&1; then
  grep -E "^error" -A12 $O/build.log | head -60 >> $O/build.txt
  echo "BUILD-FAIL" >> $O/build.txt; echo "R1X-DONE $(date +%T)" >> $O/build.txt; exit 1
fi
for b in dsv41-gate dsv41-run; do cp -a "$R/target/release/$b" "$R/bins/r1x/"; done
echo "build end $(date +%T)" >> $O/build.txt
sha256sum $R/bins/r1x/* >> $O/build.txt
export MEMRA_FULL_PREC=1
cmpset() {  # NAME SETDIR REFDIR -> "compared, differ" + the list
  local n=0 bad=0
  : > $O/diff-$1.txt
  for f in $(cd $3 && find . -name '*.bin' | sort); do
    n=$((n+1)); cmp -s $3/$f $2/$f || { bad=$((bad+1)); echo "DIFF $f" >> $O/diff-$1.txt; }
  done
  echo "$1: $n compared, $bad differ" >> $O/build.txt
}
for arm in none router comp; do
  el=$(( $(date +%s) - T0 ))
  if [ $el -gt 1260 ]; then echo "arm $arm skipped: ${el}s used" >> $O/build.txt; continue; fi
  rm -rf $R/save-r1x-$arm; mkdir -p $R/save-r1x-$arm
  MEMRA_R1X_SEG=$arm $R/bins/r1x/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 \
    --token-map $TM --floor $D/captures-v2/needle-floor --save $R/save-r1x-$arm --out $O/layer-$arm.json \
    > $O/layer-$arm.log 2>&1
  echo "layer $arm rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/layer-$arm.log | tail -1) $(date +%T)" >> $O/build.txt
  cmpset $arm-vs-k3 $R/save-r1x-$arm $R/save-k3
  cmpset $arm-vs-r1 $R/save-r1x-$arm $R/save-r1
done
echo "R1X-DONE $(date +%T)" >> $O/build.txt
