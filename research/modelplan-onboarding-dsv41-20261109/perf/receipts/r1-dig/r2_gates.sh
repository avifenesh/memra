#!/usr/bin/env bash
# GB10 (b23d, under one sparklock lease): r2 = the router branch with the fp32 FMA family back to one chain an output
# (kpc's bytes) and the narrow output's chain streamed (the ring kernel, the compressor's pair in one launch). PART
#   a: build for 121a, the kernel gate, the layer gate's saved tensors against k3's set (want 111/111 equal);
#   b: tp at 2 and 4 ranks of the card (peer-pull), decode on three captures.
# Out: out/gates-r2/. usage: r2_gates.sh a|b
set -u
PART=$1
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/gates-r2
B=$R/bins/r2
mkdir -p $O $B
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines $(date +%T)" >> $O/gates-$PART.txt; }
: > $O/gates-$PART.txt
export MEMRA_FULL_PREC=1
if [ "$PART" = a ]; then
  cd $R/src || exit 1
  tar -x -f "$R/r2-overlay.tar"
  touch crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/dsv41_gpu.rs crates/memra-engine/src/dsv41_gpu/tp.rs \
    crates/memra-engine/src/dsv41_gpu/decode.rs crates/memra-engine/src/bin/dsv41_gate.rs \
    crates/memra-engine/src/bin/dsv41_run.rs crates/memra-engine/src/bin/dsv41_tpbench.rs
  echo "build start $(date +%T)" >> $O/gates-a.txt
  if ! nice -n 19 cargo build --release --locked -j 10 -p memra-engine --bin dsv41-gate --bin dsv41-run --bin dsv41-tpbench \
      --bin dsv41-bench > $O/build.log 2>&1; then
    grep -E "^error" -A12 $O/build.log | head -60 >> $O/gates-a.txt
    echo "BUILD-FAIL" >> $O/gates-a.txt; echo "GATES-R2-DONE a $(date +%T)" >> $O/gates-a.txt; exit 1
  fi
  for b in dsv41-gate dsv41-run dsv41-tpbench dsv41-bench; do cp -a "$R/target/release/$b" "$B/"; done
  echo "build end $(date +%T)" >> $O/gates-a.txt
  sha256sum $B/* >> $O/gates-a.txt
  $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
  say "kernels" $? $O/kernels.log
  grep -a "^FAIL" $O/kernels.log | head -10 >> $O/gates-a.txt
  grep -a "gemm_ffma" $O/kernels.log >> $O/gates-a.txt
  rm -rf $R/save-r2; mkdir -p $R/save-r2
  $B/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
    --floor $D/captures-v2/needle-floor --save $R/save-r2 --out $O/layer.json > $O/layer.log 2>&1
  say "layer" $? $O/layer.log
  grep -a "^FAIL" $O/layer.log | head -10 >> $O/gates-a.txt
  n=0; bad=0; : > $O/diff-k3.txt
  for f in $(cd $R/save-k3 && find . -name '*.bin' | sort); do
    n=$((n+1)); cmp -s $R/save-k3/$f $R/save-r2/$f || { bad=$((bad+1)); echo "DIFF $f" >> $O/diff-k3.txt; }
  done
  echo "layer saved tensors against k3's set: $n compared, $bad differ" >> $O/gates-a.txt
  (cd $R/save-r2 && find . -name '*.bin' | sort | xargs sha256sum) > $O/save-r2.sha256
  sha256sum $O/save-r2.sha256 >> $O/gates-a.txt
  echo "GATES-R2-DONE a $(date +%T)" >> $O/gates-a.txt
else
  for dev in 0,0 0,0,0,0; do
    MEMRA_TP_TRANSPORT=peer-pull $B/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices $dev \
      --decode 2 --token-map $TM --out $O/tp-$dev.json > $O/tp-$dev.log 2>&1
    say "tp $dev peer-pull" $? $O/tp-$dev.log
  done
  $B/dsv41-gate decode $C $D/dec/captures/straddle126 --layers 0,1,2,3,8,14,20,21,24 --token-map $TM \
    --out $O/decode-straddle126.json > $O/decode-straddle126.log 2>&1
  say "decode straddle126" $? $O/decode-straddle126.log
  $B/dsv41-gate decode $C $D/dec/captures/gsm --layers 0,1,2,3,8,14,20,21,24 --token-map $TM \
    --out $O/decode-gsm.json > $O/decode-gsm.log 2>&1
  say "decode gsm" $? $O/decode-gsm.log
  $B/dsv41-gate decode $C $D/captures-v2/capital --layers 1,2,3,14,20,21,24 --token-map $TM \
    --out $O/decode-capital.json > $O/decode-capital.log 2>&1
  say "decode capital" $? $O/decode-capital.log
  echo "GATES-R2-DONE b $(date +%T)" >> $O/gates-b.txt
fi
