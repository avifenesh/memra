#!/usr/bin/env bash
# q-s2za.sh (second SE pair): the fused MoE bench's decomposition ($1): streaming only, compute
# only, with and without the x mirror, rows 1 and 4.
set -u
BSHA=$1
S=/root/rcpt/q-s2zc.summary; R=/root/rcpt/moe-bench4-s2zc; mkdir -p $R
export PATH=/usr/local/cuda/bin:$PATH
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-bw-bench-20260927
git -C /root/lane/t-bench checkout -q --detach $BSHA
( cd /root/lane/t-bench && nvcc -t 8 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off -arch=sm_120a -lineinfo tools/dsv4-moe-fused-bench.cu -lcublasLt -lcublas -ldl -o $R/moe_bench > $R/build.log 2>&1 )
echo "build rc=$? $(sha256sum $R/moe_bench | cut -c1-16)" >> $S
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
for rows in 1 4; do $R/moe_bench 200 3 $rows variants 9>&- > $R/variants-r$rows.log 2>&1; echo "variants rows=$rows $(grep -hE 'PACKED_EQUAL|VARIANT (std|down)' $R/variants-r$rows.log | tr '\n' ' ')" >> $S; done
rm -f $R/moe_bench
echo "S2ZC_DONE $BSHA" >> $S
