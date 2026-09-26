#!/usr/bin/env bash
# q-pair21.sh (WS pod): memra #710 lane $1: the dense-fast rows unit tests (FP8 + dots) and
# dsv4_rows_gate on TP/EP (eager, replay re-entry, graph, sampled graph, timing).
while ! grep -q PAIR20_DONE /root/rcpt/q-pair20.summary 2>/dev/null; do sleep 30; done
set -u
SHA=$1
S=/root/rcpt/q-pair21.summary; R=/root/rcpt/rows-hoist-$SHA; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-graph-20260926
git -C /root/lane/t-rg checkout -q --detach $SHA
( cd /root/lane/t-rg && CARGO_TARGET_DIR=/root/lane/target-rg cargo build --release -j 56 -p memra-engine --bin dsv4_rows_gate && \
  CARGO_TARGET_DIR=/root/lane/target-rg cargo test --release -j 56 -p memra-engine --test dsv4_dense_fast_rows_gpu --no-run ) > $R/build.log 2>&1
echo "build $SHA rc=$?" >> $S
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
wait_lock
( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; cd /root/lane/t-rg && env NVIDIA_TF32_OVERRIDE=0 CARGO_TARGET_DIR=/root/lane/target-rg cargo test --release -p memra-engine --test dsv4_dense_fast_rows_gpu -- --ignored --test-threads=1 9>&- > $R/unit.log 2>&1 )
echo "unit rc=$? $(grep -hE 'cases bit-identical|red arm|test result|panicked' $R/unit.log | cut -c1-160 | tr '\n' ' ')" >> $S
wait_lock
( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75
  env NVIDIA_TF32_OVERRIDE=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep timeout 3000 /root/lane/target-rg/release/dsv4_rows_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64 9>&- > $R/gate.log 2>&1 )
echo "rows-graph rc=$? $(grep -hE 'PASS|FAILED|DIVERGENCE|TIME rep=1|panicked' $R/gate.log | cut -c1-220 | tr '\n' ' ')" >> $S
echo "PAIR21_DONE $SHA" >> $S
