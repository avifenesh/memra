#!/bin/bash
# q-mb1.sh (1x PRO 6000): indexer dispatch-knee sweep (#451), then the HC2 bit gate x2 and timing (#19).
S=/root/rcpt/q-mb1.summary; mkdir -p /root/rcpt/idx /root/rcpt/hc2
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv,noheader >> $S
cd /root/memra && git fetch -q origin lane/dsv4-indexer-door-451-20260923 lane/dsv4-mhc-20260923
git worktree add -f /root/idx origin/lane/dsv4-indexer-door-451-20260923 > /dev/null 2>&1
git worktree add -f /root/mhc origin/lane/dsv4-mhc-20260923 > /dev/null 2>&1
echo "idx-tree $(git -C /root/idx rev-parse HEAD) mhc-tree $(git -C /root/mhc rev-parse HEAD)" >> $S
# HC2 test binary builds in the background while the knee runs.
( cd /root/mhc && CARGO_TARGET_DIR=/root/target cargo test --release -j 30 -p memra-engine --test dsv4_hc_finish_gpu --no-run > /root/rcpt/hc2/build.log 2>&1; echo "hc2 build rc=$?" >> $S ) &
cd /root/idx && nvcc -O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-indexer-tiled-gate.cu -lcublasLt -o /root/idx-gate > /root/rcpt/idx/nvcc.log 2>&1
echo "idx nvcc rc=$? $(sha256sum /root/idx-gate | cut -c1-16)" >> $S
flock /tmp/memra-gpu.lock /root/idx-gate > /root/rcpt/idx/gate.log 2>&1; echo "idx gate rc=$? $(tail -n 1 /root/rcpt/idx/gate.log)" >> $S
flock /tmp/memra-gpu.lock /root/idx-gate --teeth > /root/rcpt/idx/teeth.log 2>&1; echo "idx teeth rc=$? $(tail -n 1 /root/rcpt/idx/teeth.log)" >> $S
for r in 1 2; do
  flock /tmp/memra-gpu.lock /root/idx-gate --knee > /root/rcpt/idx/knee-r$r.log 2>&1; echo "idx knee r$r rc=$? $(tail -n 1 /root/rcpt/idx/knee-r$r.log)" >> $S
done
wait
T=$(grep -o '/root/target/release/deps/dsv4_hc_finish_gpu-[0-9a-f]*' /root/rcpt/hc2/build.log | tail -1)
echo "hc2 test-bin $(sha256sum $T | cut -c1-16)" >> $S
for r in 1 2; do
  NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-gpu.lock $T --ignored --nocapture --test-threads=1 dsv4_hc_finish_is_bit_identical > /root/rcpt/hc2/gate-r$r.log 2>&1
  echo "hc2 gate r$r rc=$? $(grep -hE 'EXACT|test result|panicked' /root/rcpt/hc2/gate-r$r.log | tr '\n' ' ' | cut -c1-400)" >> $S
done
NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-gpu.lock $T --ignored --nocapture --test-threads=1 dsv4_hc_finish_timing > /root/rcpt/hc2/timing.log 2>&1
echo "hc2 timing rc=$?" >> $S; grep TIMING /root/rcpt/hc2/timing.log >> $S
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader >> $S
echo MB1_DONE >> $S
