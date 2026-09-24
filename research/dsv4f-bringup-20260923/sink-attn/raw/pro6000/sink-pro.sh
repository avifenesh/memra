#!/usr/bin/env bash
# sink-pro.sh (1x PRO 6000 microbench box): dsv4_sink_attn_st_gpu on the committed lane tip.
S=/root/sink/summary; mkdir -p /root/sink; exec > /root/sink/run.log 2>&1
[[ -x /root/.cargo/bin/cargo ]] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. /root/.cargo/env
[[ -d /root/memra ]] || git clone -q https://github.com/avifenesh/memra /root/memra
cd /root/memra && git fetch -q origin lane/dsv4-sink-attn-20260923 && git checkout -q FETCH_HEAD
echo "tree $(git rev-parse HEAD)" >> $S
export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cargo test --release -j 96 -p memra-engine --test dsv4_sink_attn_st_gpu --no-run > /root/sink/build.log 2>&1
echo "build rc=$? $(tail -n 3 /root/sink/build.log | tr '\n' ' ')" >> $S
T=$(grep -o 'target/release/deps/dsv4_sink_attn_st_gpu-[0-9a-f]*' /root/sink/build.log | tail -1)
echo "test-bin $T $(sha256sum $T | cut -c1-16)" >> $S
nvidia-smi --query-gpu=name,power.limit,temperature.gpu,clocks.max.sm --format=csv,noheader >> $S
for r in 1 2 3; do
  DSV4_LATENCY_TREE=lane NVIDIA_TF32_OVERRIDE=0 $T --ignored --nocapture --test-threads=1 > /root/sink/gate-r$r.log 2>&1
  echo "gate r$r rc=$? $(grep -hE 'cases|red arm|test result' /root/sink/gate-r$r.log | tr '\n' ' ')" >> $S
done
echo SINKPRO_DONE >> $S
