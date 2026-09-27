#!/usr/bin/env bash
# q-v6f.sh (SE pair): Nsight Compute on the exact prefill FP8 tile (single card, the kernel-boundary
# bit test dsv4_gemm_tile_gpu, which runs the served prefill shapes). The two-card long gate cannot
# run under ncu: ncu serializes kernels, so a cross-rank join spin times out (40043).
set -u
S=/root/rcpt/q-v6f.summary; R=/root/rcpt/ncu-tile-v6f; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/t-main13 && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --test dsv4_gemm_tile_gpu --no-run -j 56 > $R/build.log 2>&1
TB=$(grep -oE '/root/lane/target-main13/release/deps/dsv4_gemm_tile_gpu-[0-9a-f]+' $R/build.log | head -1)
echo "test binary $TB" >> $S
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
env CUDA_VISIBLE_DEVICES=0 timeout 2400 ncu --kernel-name regex:dsv4_gemm_fp8_tile --launch-count 12 --set full --import-source no -f -o $R/tile $TB --ignored --test-threads=1 9>&- > $R/ncu.log 2>&1
echo "ncu rc=$? $(grep -c 'Profiling' $R/ncu.log) kernels" >> $S
ncu --import $R/tile.ncu-rep --page raw --csv --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_eligible.avg.per_cycle_active > $R/metrics.csv 2>&1
ncu --import $R/tile.ncu-rep --page details --csv --section WarpStateStats --section ComputeWorkloadAnalysis --section Occupancy > $R/details.csv 2>&1
echo V6F_DONE >> $S
