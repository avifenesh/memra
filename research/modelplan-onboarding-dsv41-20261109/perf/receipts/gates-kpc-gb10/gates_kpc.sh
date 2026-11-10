#!/usr/bin/env bash
# GB10 (under the units' shared lease): k3 + the transport fix (kpc = the lane's two commits), built for 121a, and its
# one-card tp gates: 2 and 4 ranks on the card on peer-pull, the default transport and host-canonical, the graph
# door on 2 ranks; then the layer gate's saved tensors against k3's set (the fix must not move a byte).
set -u
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/gates-kpc
mkdir -p $O $R/bins/kpc
: > $O/gates.txt
cd $R/src || exit 1
tar -x -f "$R/k3-overlay.tar"
tar -x -f "$R/pc1-overlay.tar"
touch crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/bin/dsv41_gate.rs crates/memra-engine/src/dsv41_gpu/tp.rs
echo "build start $(date +%T)" >> $O/gates.txt
if ! nice -n 19 cargo build --release --locked -j 10 -p memra-engine --bin dsv41-gate --bin dsv41-run --bin dsv41-tpbench \
    --bin dsv41-bench > $O/build.log 2>&1; then
  echo "BUILD-FAIL" >> $O/gates.txt; echo "GATES-KPC-DONE $(date +%T)" >> $O/gates.txt; exit 1
fi
for b in dsv41-gate dsv41-run dsv41-tpbench dsv41-bench; do cp -a "$R/target/release/$b" "$R/bins/kpc/"; done
sha256sum $R/bins/kpc/* >> $O/gates.txt
export MEMRA_FULL_PREC=1
B=$R/bins/kpc
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/gates.txt; }
for dev in 0,0 0,0,0,0; do
  for tr in peer-pull default host-canonical; do
    if [ $tr = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=$tr; fi
    $B/dsv41-gate tp $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --devices $dev --decode 2 --token-map $TM \
      --out $O/tp-$dev-$tr.json > $O/tp-$dev-$tr.log 2>&1
    say "tp $dev $tr" $? $O/tp-$dev-$tr.log
    grep -a "stress/" $O/tp-$dev-$tr.log | cut -c1-200 >> $O/gates.txt
  done
done
unset MEMRA_TP_TRANSPORT
MEMRA_DSV41_DECODE_GRAPH=1 MEMRA_TP_TRANSPORT=device-push $B/dsv41-gate tp $C $D/captures-v2/needle \
  --layers 1,2,14,20,21,24 --devices 0,0 --decode 2 --token-map $TM --out $O/tp-graph.json > $O/tp-graph.log 2>&1
say "tp 0,0 graph door" $? $O/tp-graph.log
grep -a "graph/replayed" $O/tp-graph.log | cut -c1-200 >> $O/gates.txt
rm -rf $R/save-kpc; mkdir -p $R/save-kpc
$B/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
  --floor $D/captures-v2/needle-floor --save $R/save-kpc --out $O/layer.json > $O/layer.log 2>&1
say "layer" $? $O/layer.log
n=0; bad=0
for f in $(cd $R/save-k3 && find . -name '*.bin' | sort); do
  n=$((n+1)); cmp -s $R/save-k3/$f $R/save-kpc/$f || bad=$((bad+1))
done
echo "layer saved tensors against k3's set: $n compared, $bad differ" >> $O/gates.txt
rm -rf $R/save-kpc
echo "GATES-KPC-DONE $(date +%T)" >> $O/gates.txt
