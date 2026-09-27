#!/usr/bin/env bash
# q-v6b.sh (SE pair): Nsight Compute on the small decode kernels of the TP/EP step (fused MoE lane
# binary, PDL off): duration, throughput, occupancy and warp-stall reasons for the latency-bound
# chain (HC finish, the q norm pack, the router, sink attention, the indexer, HC split dots).
set -u
S=/root/rcpt/q-v6b.summary; R=/root/rcpt/ncu-small-v6b; mkdir -p $R
until grep -q V6A_DONE /root/rcpt/q-v6a.summary 2>/dev/null; do sleep 60; done
export PATH=/usr/local/cuda/bin:$PATH
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt; X=/root/lane/bin-drowsZ/dsv4_tp_replay_long_gate
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
KR='regex:dsv4_hc_finish|dsv4_small_norm_pack|dsv4_route_m|dsv4_sink_scores_st|dsv4_sink_softout_st|dsv4_indexer_score_f32acc_kernel|dsv4_topk_idx_numeric|dsv4_hc_dot_split_partial|dsv4_moe_fused_gu|dsv4_moe_fused_down|dsv4_dense_fast_fp8'
env MEMRA_DSV4_PDL=0 NVIDIA_TF32_OVERRIDE=0 timeout 3000 ncu --target-processes all --kernel-name "$KR" --launch-skip 4000 --launch-count 120 \
  --set full --import-source no -f -o $R/small $X $M $T 64 9>&- > $R/ncu.log 2>&1
echo "ncu rc=$? $(tail -2 $R/ncu.log | tr '\n' ' ' | cut -c1-200)" >> $S
ncu --import $R/small.ncu-rep --page raw --csv --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread > $R/metrics.csv 2>&1
ncu --import $R/small.ncu-rep --page details --csv --section WarpStateStats > $R/warpstate.csv 2>&1
echo V6B_DONE >> $S
