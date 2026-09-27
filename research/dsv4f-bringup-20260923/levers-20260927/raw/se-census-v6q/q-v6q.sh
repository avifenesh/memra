#!/usr/bin/env bash
# q-v6q.sh (SE pair): the step census on the HC-warp lane's long gate (bin-W): nsys graph-node
# tracing of the replayed steps, PDL off (clean per-kernel durations) and on (the served chain).
set -u
S=/root/rcpt/q-v6q.summary; R=/root/rcpt/census-v6q; mkdir -p $R
until grep -q V6P_DONE /root/rcpt/q-v6p.summary 2>/dev/null; do sleep 60; done
export PATH=/usr/local/cuda/bin:$PATH
X=/root/lane/bin-W/dsv4_tp_replay_long_gate; sha256sum $X > $R/binary.sha256
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
for pdl in 0 1; do
  d=$R/replay-pdl$pdl; mkdir -p $d
  ( NVIDIA_TF32_OVERRIDE=0 MEMRA_DSV4_PDL=$pdl DSV4_REPLAY_GATE_PROFILE=replay nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --cuda-graph-trace=node -o $d/nsys -f true $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 304 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 304 > $d/ana.txt 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/nsys.nsys-rep > /dev/null 2>&1
  rm -f $d/nsys.sqlite $d/nsys.nsys-rep
  echo "census pdl$pdl $(grep -hE 'PASS:|PROFILE|FAILED|panicked' $d/gate.log | cut -c1-120 | tr '\n' ' ') | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
echo "V6Q_DONE" >> $S
