#!/usr/bin/env bash
# GB10 lease A (<= 30 min): emb = kpc (k3 + the transport fix) + the embedding table on every rank with the row
# gather on the card, Engram's staging pinned at load, the runs' reserve mapped into the pool at load. Built for 121a:
# fmt, build, the kernel gate (two new checks: schedule/embed_rows), the layer gate's saved tensors against k3's set
# (nothing of the per-layer program changes), and the pool probe (does a reserved, freed block serve later
# allocations of other sizes without the pool growing).
set -u
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/gates-emb
mkdir -p $O $R/bins/emb
: > $O/gates-a.txt
cd $R/src || exit 1
tar -x -f "$R/emb-overlay.tar"
touch crates/memra-engine/cu/dsv41_gpu.cu crates/memra-engine/src/dsv41_gpu.rs crates/memra-engine/src/dsv41_gpu/tp.rs \
  crates/memra-engine/src/dsv41_gpu/decode.rs crates/memra-engine/src/bin/dsv41_gate.rs \
  crates/memra-engine/src/bin/dsv41_run.rs crates/memra-engine/src/bin/dsv41_tpbench.rs
nice -n 19 cargo fmt -p memra-engine -- --check > $O/fmt.log 2>&1
echo "fmt rc $?" >> $O/gates-a.txt
echo "build start $(date +%T)" >> $O/gates-a.txt
if ! nice -n 19 timeout 15m cargo build --release --locked -j 10 -p memra-engine --bin dsv41-gate --bin dsv41-run \
    --bin dsv41-tpbench --bin dsv41-bench > $O/build.log 2>&1; then
  grep -E "^error" -A14 $O/build.log | head -80 >> $O/gates-a.txt
  echo "BUILD-FAIL" >> $O/gates-a.txt; echo "GATES-EMB-A-DONE $(date +%T)" >> $O/gates-a.txt; exit 1
fi
echo "build end $(date +%T)" >> $O/gates-a.txt
for b in dsv41-gate dsv41-run dsv41-tpbench dsv41-bench; do cp -a "$R/target/release/$b" "$R/bins/emb/"; done
sha256sum $R/bins/emb/* >> $O/gates-a.txt
export MEMRA_FULL_PREC=1
B=$R/bins/emb
say() { echo "$1 rc $2 $(grep -o '[0-9]* checks, [0-9]* failed' $3 | tail -1) $(grep -ac '^FAIL' $3) FAIL lines" >> $O/gates-a.txt; }
timeout 5m $B/dsv41-gate kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
say "kernels" $? $O/kernels.log
grep -a "^FAIL" $O/kernels.log | head -10 >> $O/gates-a.txt
grep -a "embed_rows" $O/kernels.log | cut -c1-200 >> $O/gates-a.txt
timeout 3m python3 -I $R/pool_probe.py 0 > $O/pool-probe.txt 2>&1
echo "pool probe rc $?" >> $O/gates-a.txt
cat $O/pool-probe.txt >> $O/gates-a.txt
rm -rf $R/save-emb; mkdir -p $R/save-emb
timeout 10m $B/dsv41-gate layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
  --floor $D/captures-v2/needle-floor --save $R/save-emb --out $O/layer.json > $O/layer.log 2>&1
say "layer" $? $O/layer.log
n=0; bad=0
for f in $(cd $R/save-k3 && find . -name '*.bin' | sort); do
  n=$((n+1)); cmp -s $R/save-k3/$f $R/save-emb/$f || bad=$((bad+1))
done
echo "layer saved tensors against k3's set: $n compared, $bad differ" >> $O/gates-a.txt
rm -rf $R/save-emb
echo "GATES-EMB-A-DONE $(date +%T)" >> $O/gates-a.txt
