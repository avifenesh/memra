#!/usr/bin/env bash
# GB10, one build lease: bs1f = bs1 with the small-row ring sized by the call (6 stages) + the gate with the tp hash log and the partial-N schedule cases
# (n = 576, 1,344: a last 128-column tile half full; dense and tile list). dsv41-gate only.
set -u
R=$HOME/.cache/dsv41-bsmma
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/build-bs1f; mkdir -p $O $R/bins/bs1f; : > $O/runs.txt
E=$R/src/crates/memra-engine
cd $R/src || exit 1
cp $R/dsv41_gate_e.rs $E/src/bin/dsv41_gate.rs
cp $R/bs1f_dsv41_gpu.cu $E/cu/dsv41_gpu.cu
touch $E/src/bin/dsv41_gate.rs $E/cu/dsv41_gpu.cu
nice -n 19 timeout 20m cargo build --release --locked -j 10 -p memra-engine --bin dsv41-gate --bin dsv41-bench --bin dsv41-run --bin dsv41-tpbench > $O/build.log 2>&1
rc=$?
echo "bs1f build rc $rc $(date +%T)" >> $O/runs.txt
[ $rc -eq 0 ] && for b in dsv41-gate dsv41-bench dsv41-run dsv41-tpbench; do cp -a $R/target/release/$b $R/bins/bs1f/; done && sha256sum $R/bins/bs1f/dsv41-gate >> $O/runs.txt
[ $rc -ne 0 ] && grep -E "^error" -A10 $O/build.log | head -30 >> $O/runs.txt
echo "BUILD-BS1F-DONE $(date +%T)" >> $O/runs.txt
