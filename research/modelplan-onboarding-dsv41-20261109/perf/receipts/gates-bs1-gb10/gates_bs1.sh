#!/usr/bin/env bash
# GB10 (spark-b23d, under a sparklock lease): bs1 = k3 (3ae95d01cf) + the FP8/FP4 GEMM family on the hardware
# block-scaled MMA (one chain per output over ascending K inside the tensor core), built for 121a, then its one-card
# gates: kernels (tile == small-row bytes, CPU reference bands), the layer gate saving the new reference set (moved
# tensors counted against k3's set), tp at 2 and 4 ranks on the card on three transports, decode on three captures.
# Reads the speed lane's data, k3 set and nothing else of it; writes only under ~/.cache/dsv41-bsmma.
set -u
R=$HOME/.cache/dsv41-bsmma
S=$HOME/.cache/dsv41-speed
D=$S/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/gates-bs1
mkdir -p $O $R/bins/bs1 $R/src
: > $O/gates.txt
cd $R/src || exit 1
tar -x -f "$R/bs-src.tar"
touch crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/dsv41_gpu.rs crates/memra-engine/src/bin/dsv41_gate.rs
# first: the standalone layout probe of the block-scaled MMA (fragments, scale selectors, exchanged operands,
# e2m1 containers); the full kernels' gates run only if it passes
mkdir -p $O/probe
/usr/local/cuda/bin/nvcc -O2 -gencode arch=compute_121a,code=sm_121a -o $O/probe/bsprobe121 $R/bsprobe.cu > $O/probe/nvcc.log 2>&1
echo "probe nvcc rc $?" >> $O/gates.txt
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader > $O/probe/probe-out.txt 2>&1
timeout 120 $O/probe/bsprobe121 >> $O/probe/probe-out.txt 2>&1
PRC=$?
echo "probe rc $PRC" >> $O/gates.txt
cat $O/probe/probe-out.txt >> $O/gates.txt
if [ $PRC -ne 0 ]; then echo "PROBE-FAIL: gates not run" >> $O/gates.txt; echo "GATES-BS1-DONE $(date +%T)" >> $O/gates.txt; exit 1; fi
echo "build start $(date +%T)" >> $O/gates.txt
if ! nice -n 19 cargo build --release --locked -j 10 -p memra-engine --bin dsv41-gate --bin dsv41-run --bin dsv41-tpbench \
    --bin dsv41-bench > $O/build.log 2>&1; then
  grep -E "^error" -A12 $O/build.log | head -60 >> $O/gates.txt
  echo "BUILD-FAIL" >> $O/gates.txt; echo "GATES-BS1-DONE $(date +%T)" >> $O/gates.txt; exit 1
fi
echo "build end $(date +%T)" >> $O/gates.txt
for b in dsv41-gate dsv41-run dsv41-tpbench dsv41-bench; do cp -a "$R/target/release/$b" "$R/bins/bs1/"; done
# the onboarding CLI for CheckpointParity in the tapes lease (not part of these gates; a failure is noted only)
nice -n 19 cargo build --release --locked -j 10 -p memra-cli --bin memra > $O/build-cli.log 2>&1
echo "memra-cli build rc $?" >> $O/gates.txt
[ -x "$R/target/release/memra" ] && cp -a "$R/target/release/memra" "$R/bins/bs1/"
sha256sum $R/bins/bs1/* >> $O/gates.txt
export MEMRA_FULL_PREC=1
B=$R/bins/bs1
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/gates.txt; }
$B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
say "kernels" $? $O/kernels.log
grep -a "^FAIL" $O/kernels.log | head -20 >> $O/gates.txt
rm -rf $R/save-bs1; mkdir -p $R/save-bs1
$B/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
  --floor $D/captures-v2/needle-floor --save $R/save-bs1 --out $O/layer.json > $O/layer.log 2>&1
say "layer" $? $O/layer.log
grep -a "^FAIL" $O/layer.log | head -10 >> $O/gates.txt
n=0; bad=0; : > $O/diff-k3.txt
for f in $(cd $S/save-k3 && find . -name '*.bin' | sort); do
  n=$((n+1)); cmp -s $S/save-k3/$f $R/save-bs1/$f || { bad=$((bad+1)); echo "DIFF $f" >> $O/diff-k3.txt; }
done
echo "layer saved tensors against k3's set: $n compared, $bad differ" >> $O/gates.txt
(cd $R/save-bs1 && find . -name '*.bin' | sort | xargs sha256sum) > $O/save-bs1.sha256
sha256sum $O/save-bs1.sha256 >> $O/gates.txt
for tr in peer-pull host-canonical default; do
  for dev in 0,0 0,0,0,0; do
    if [ $tr = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=$tr; fi
    $B/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices $dev \
      --decode 2 --token-map $TM --out $O/tp-$tr-$dev.json > $O/tp-$tr-$dev.log 2>&1
    say "tp $dev $tr" $? $O/tp-$tr-$dev.log
    grep -a "^FAIL" $O/tp-$tr-$dev.log | head -5 >> $O/gates.txt
  done
done
unset MEMRA_TP_TRANSPORT
$B/dsv41-gate decode $C $D/dec/captures/straddle126 --layers 0,1,2,3,8,14,20,21,24 --token-map $TM \
  --out $O/decode-straddle126.json > $O/decode-straddle126.log 2>&1
say "decode straddle126" $? $O/decode-straddle126.log
$B/dsv41-gate decode $C $D/dec/captures/gsm --layers 0,1,2,3,8,14,20,21,24 --token-map $TM \
  --out $O/decode-gsm.json > $O/decode-gsm.log 2>&1
say "decode gsm" $? $O/decode-gsm.log
$B/dsv41-gate decode $C $D/captures-v2/capital --layers 1,2,3,14,20,21,24 --token-map $TM \
  --out $O/decode-capital.json > $O/decode-capital.log 2>&1
say "decode capital" $? $O/decode-capital.log
echo "GATES-BS1-DONE $(date +%T)" >> $O/gates.txt
