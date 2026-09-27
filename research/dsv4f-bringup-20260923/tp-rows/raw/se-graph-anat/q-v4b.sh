#!/usr/bin/env bash
# q-v4b.sh (SE pair): memra #710 anatomy of the captured TP/EP B-row step, lane 3d3e8936d.
# nsys with graph-node tracing over 64 graph steps at B=2 and B=4.
while ! grep -q V4A_DONE /root/rcpt/q-v4a.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v4b.summary; R=/root/rcpt/rows-graph-anat; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-graph-20260926
git worktree add -f /root/lane/t-rg 3d3e8936d > /dev/null 2>&1; git -C /root/lane/t-rg checkout -q --detach 3d3e8936d
[[ -d /root/lane/target-rg ]] || cp -a /root/lane/target-rows2 /root/lane/target-rg
( cd /root/lane/t-rg && CARGO_TARGET_DIR=/root/lane/target-rg cargo build --release -j 56 -p memra-engine --bin dsv4_rows_gate ) > $R/build.log 2>&1
X=/root/lane/target-rg/release/dsv4_rows_gate
echo "build $(git -C /root/lane/t-rg rev-parse --short HEAD) rc=$?" >> $S
exec 9>/tmp/memra-gpu.lock
while ! flock -n 9; do sleep 10; done
for b in 2 4; do
  d=$R/b$b; mkdir -p $d
  ( NVIDIA_TF32_OVERRIDE=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_PROFILE=$b DSV4_ROWS_GATE_PROFILE_GRAPH=1 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --cuda-graph-trace=node -o $d/nsys -f true $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 64 > $d/ana.txt 2>&1
  rm -f $d/nsys.sqlite $d/nsys.nsys-rep
  echo "anat graph B=$b $(grep -hE 'PROFILE|panicked' $d/gate.log | cut -c1-100) | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
echo V4B_DONE >> $S
