#!/usr/bin/env bash
# q-v6r.sh (SE pair): the B-row step census, the rows gate's profile mode at B=4 on the captured
# TP/EP B-row step (bin-L), nsys graph-node tracing, PDL off.
set -u
S=/root/rcpt/q-v6r.summary; R=/root/rcpt/census-rows-v6r; mkdir -p $R
until grep -q V6Q_DONE /root/rcpt/q-v6q.summary 2>/dev/null; do sleep 60; done
export PATH=/usr/local/cuda/bin:$PATH
X=/root/lane/bin-L/dsv4_rows_gate; sha256sum $X > $R/binary.sha256
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
for b in 1 4; do
  d=$R/rows-b$b; mkdir -p $d
  ( NVIDIA_TF32_OVERRIDE=0 MEMRA_DSV4_PDL=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_PROFILE=$b DSV4_ROWS_GATE_PROFILE_GRAPH=1 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --cuda-graph-trace=node -o $d/nsys -f true $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 64 > $d/ana.txt 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/nsys.nsys-rep > /dev/null 2>&1
  rm -f $d/nsys.sqlite $d/nsys.nsys-rep
  echo "rows census b$b $(grep -hE 'PROFILE|FAILED|panicked' $d/gate.log | cut -c1-120 | tr '\n' ' ') | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
echo "V6R_DONE" >> $S
