#!/usr/bin/env bash
# q-v3w.sh (2x RTX PRO 6000 Server Edition): memra #710 anatomy of the replayed TP/EP step on
# the served stream MoE with the served checks on (anatomy lane 5a371c847 = profile mode + the
# capacity/greedy/checks lane). nsys (cudaProfilerApi, graph-node tracing) over one run of 304
# steps, arm=replay then arm=eager.
set -u
S=/root/rcpt/q-v3w.summary; R=/root/rcpt/tpanat2; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-anatomy-20260924
git worktree add -f /root/lane/t-tpa2 5a371c847 > /dev/null 2>&1; git -C /root/lane/t-tpa2 checkout -q --detach 5a371c847
[[ -d /root/lane/target-tpa2 ]] || cp -a /root/lane/target-tpv /root/lane/target-tpa2
( cd /root/lane/t-tpa2 && CARGO_TARGET_DIR=/root/lane/target-tpa2 cargo build --release -j 44 -p memra-engine --bin dsv4_tp_replay_long_gate ) > $R/build.log 2>&1
X=/root/lane/target-tpa2/release/dsv4_tp_replay_long_gate
echo "build tpa $(git -C /root/lane/t-tpa2 rev-parse --short HEAD) gate $(sha256sum $X | cut -c1-16)" >> $S
exec 9>/tmp/memra-gpu.lock
while ! flock -n 9; do sleep 10; done
for arm in replay eager; do
  d=$R/$arm; mkdir -p $d
  ( NVIDIA_TF32_OVERRIDE=0 DSV4_REPLAY_GATE_PROFILE=$arm nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --cuda-graph-trace=node -o $d/nsys -f true $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 304 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 304 > $d/ana.txt 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/nsys.nsys-rep > /dev/null 2>&1
  rm -f $d/nsys.sqlite
  echo "anat $arm $(grep -hE 'PASS:|PROFILE|FAILED|panicked' $d/gate.log | cut -c1-160 | tr '\n' ' ') | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
echo V3W_DONE >> $S
