#!/usr/bin/env bash
# q-v3.sh (2x RTX PRO 6000 Server Edition): exact attention TP2
# (lane/dsv4-attn-tp-exact-20260923 75c55dfd4, on main 6978f5fac) plus the lane-only
# topology selector (topology-lane.patch, never merged). Proof: the TP/EP gate DIGEST with
# attention TP off and on must be equal; the multi-row verify gate and the DSpark TP/EP gate
# on the exact program. Then the served questions: does DSpark boot on TP/EP at the served
# context now that the full attention planes are freed, and what does it serve against PP-2.
while ! grep -q SETUP_DONE /root/setup.log 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3.summary; R=/root/rcpt/attn; mkdir -p $R/plain $R/spec $R/prof
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
# First, the PP-2 request pipelining gate (memra #667, lane/dsv4-pp-pipeline-20260923 4049a5833):
# two sessions serial vs pipelined, token identity and aggregate tok/s. One load, short.
cd /root/lane/memra && git fetch -q origin lane/dsv4-pp-pipeline-20260923
git worktree add -f /root/lane/t-pipe 4049a5833 > /dev/null 2>&1
git -C /root/lane/t-pipe checkout -q --detach 4049a5833
echo "tree pipe $(git -C /root/lane/t-pipe rev-parse HEAD)" >> $S
nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.max.sm --format=csv,noheader >> $S
bash /root/box/build.sh /root/lane/t-pipe /root/lane/target-pipe pipe
P=/root/lane/target-pipe/release
echo "build pipe $(grep -hE 'EXIT' /root/build-pipe.log | tr '\n' ' ') gate $(sha256sum $P/dsv4_pipeline_gate | cut -c1-16)" >> $S
while ! grep -q DL_EXIT /root/dl2.log 2>/dev/null; do sleep 30; done
grep -q DL_EXIT=0 /root/dl2.log || { echo "DL FAILED $(tail -c 300 /root/dl2.log)" >> $S; exit 1; }
bash /root/box/hashchk.sh; echo "hashchk $(grep HASHCHK /root/hashchk.log)" >> $S
grep -q 'bad=0' /root/hashchk.log || { echo "HASH FAILED" >> $S; exit 1; }
grep -q '^PASS' /root/rcpt/pipe/gate/gate.log 2>/dev/null || { wait_lock; /root/box/gate.sh /root/rcpt/pipe/gate $P/dsv4_pipeline_gate /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 256 5; }
echo "pipe gate $(grep -hE '^(TIME|PASS|SESSION)|panicked|GATE_DONE' /root/rcpt/pipe/gate/gate.log | cut -c1-220 | tr '\n' ' ')" >> $S
# Served lanes: one binary, MEMRA_DSV4_SESSIONS 1 vs 2 (5 boots each, balanced), then 4 twice.
mkdir -p /root/rcpt/pipe/served
i=0
for arm in s2 s1 s1 s2 s2 s1 s1 s2 s2 s1 s4 s4; do
  i=$((i+1)); wait_lock; d=/root/rcpt/pipe/served/r$i-$arm
  /root/box/cell.sh $d $P/memra-server /root/box/cells-pipe.txt MEMRA_DSV4_SESSIONS=${arm#s} > $d.out 2>&1
  echo "pipe served r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-170 | tr '\n' ' ') $(grep -h FATAL $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
done
cd /root/lane/memra && git fetch -q origin lane/dsv4-attn-tp-exact-20260923
git worktree add -f /root/lane/t-attn 75c55dfd4 > /dev/null 2>&1
cd /root/lane/t-attn && git apply /root/box/topology-lane.patch
echo "tree attn $(git rev-parse HEAD) + topology-lane.patch $(sha256sum /root/box/topology-lane.patch | cut -c1-16)" >> $S
