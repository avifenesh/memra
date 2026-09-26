#!/bin/bash
# q-mb2.sh: knee rows s=16/32 plus the HC2 gate with the row red arm.
S=/root/rcpt/q-mb2.summary
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/memra && git fetch -q origin lane/dsv4-indexer-door-451-20260923 lane/dsv4-mhc-20260923
git -C /root/idx checkout -q --detach origin/lane/dsv4-indexer-door-451-20260923
git -C /root/mhc checkout -q --detach origin/lane/dsv4-mhc-20260923
echo "idx-tree $(git -C /root/idx rev-parse HEAD) mhc-tree $(git -C /root/mhc rev-parse HEAD)" >> $S
( cd /root/mhc && CARGO_TARGET_DIR=/root/target cargo test --release -j 30 -p memra-engine --test dsv4_hc_finish_gpu --no-run > /root/rcpt/hc2/build2.log 2>&1; echo "hc2 build rc=$?" >> $S ) &
cd /root/idx && nvcc -O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-indexer-tiled-gate.cu -lcublasLt -o /root/idx-gate2 > /root/rcpt/idx/nvcc2.log 2>&1
echo "idx nvcc rc=$? $(sha256sum /root/idx-gate2 | cut -c1-16)" >> $S
flock /tmp/memra-gpu.lock /root/idx-gate2 --knee > /root/rcpt/idx/knee-r3.log 2>&1; echo "idx knee r3 rc=$? $(tail -n 1 /root/rcpt/idx/knee-r3.log)" >> $S
wait
T=$(grep -o '/root/target/release/deps/dsv4_hc_finish_gpu-[0-9a-f]*' /root/rcpt/hc2/build2.log | tail -1)
echo "hc2 test-bin $(sha256sum $T | cut -c1-16)" >> $S
for r in 3 4; do
  NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-gpu.lock $T --ignored --nocapture --test-threads=1 dsv4_hc_finish_is_bit_identical > /root/rcpt/hc2/gate-r$r.log 2>&1
  echo "hc2 gate r$r rc=$? $(grep -hE 'EXACT|test result|panicked' /root/rcpt/hc2/gate-r$r.log | tr '\n' ' ' | cut -c1-400)" >> $S
done
echo MB2_DONE >> $S
