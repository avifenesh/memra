#!/usr/bin/env bash
# q-s2z.sh (second SE pair): the fused MoE pair's bandwidth on one card ($1, the bench), rows 1
# and 4 at the TP/EP partition shape, then Nsight Compute on the gate/up and down kernels.
set -u
BSHA=$1
S=/root/rcpt/q-s2z.summary; R=/root/rcpt/moe-bench-s2z; mkdir -p $R
until grep -q S2Y_DONE /root/rcpt/q-s2y.summary 2>/dev/null; do sleep 60; done
export PATH=/usr/local/cuda/bin:$PATH
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-bw-bench-20260927
git worktree add -f /root/lane/t-bench $BSHA > /dev/null 2>&1; git -C /root/lane/t-bench checkout -q --detach $BSHA
( cd /root/lane/t-bench && nvcc -t 8 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off -arch=sm_120a -lineinfo tools/dsv4-moe-fused-bench.cu -lcublasLt -lcublas -ldl -o $R/moe_bench > $R/build.log 2>&1 )
echo "build rc=$? $(sha256sum $R/moe_bench | cut -c1-16)" >> $S
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
for rep in 1 2; do for rows in 1 4; do
  $R/moe_bench 400 3 $rows 9>&- > $R/bench-r$rows-rep$rep.log 2>&1
  echo "bench rep=$rep $(grep -h BENCH $R/bench-r$rows-rep$rep.log)" >> $S
done; done
for rows in 1 4; do $R/moe_bench 200 3 $rows variants 9>&- > $R/variants-r$rows.log 2>&1; echo "variants rows=$rows $(grep -hE "PACKED_EQUAL|VARIANT" $R/variants-r$rows.log | tr "\n" " ")" >> $S; done
ncu --kernel-name regex:dsv4_moe_fused --launch-skip 40 --launch-count 4 --set full --import-source yes -f -o $R/fused $R/moe_bench 60 3 1 9>&- > $R/ncu.log 2>&1
echo "ncu rc=$? $(tail -1 $R/ncu.log | cut -c1-120)" >> $S
ncu --import $R/fused.ncu-rep --page details --csv > $R/details.csv 2>&1
ncu --import $R/fused.ncu-rep --page raw --csv --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__issue_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__grid_size > $R/metrics.csv 2>&1
echo "S2Z_DONE $BSHA" >> $S
