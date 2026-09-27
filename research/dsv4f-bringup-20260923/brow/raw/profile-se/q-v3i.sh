#!/usr/bin/env bash
# q-v3i.sh (2x RTX PRO 6000 Server Edition), after q-v3h.sh: where an added B-row costs its ~7 ms
# (memra #667 lever 2). nsys over B-row steps only (DSV4_ROWS_GATE_PROFILE, cudaProfilerApi
# capture range) at B=1, 2 and 4, lane/dsv4-brow-serve-20260924 48b2aea6d; kernel summaries per run.
while ! grep -q V3H_DONE /root/rcpt/q-v3h.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3i.summary; R=/root/rcpt/rowsprof; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-serve-20260924
git -C /root/lane/t-rows checkout -q --detach 48b2aea6d
bash /root/box/build.sh /root/lane/t-rows /root/lane/target-rows rowsprof
X=/root/lane/target-rows/release/dsv4_rows_gate
echo "build rowsprof $(grep -hE 'EXIT' /root/build-rowsprof.log | tr '\n' ' ') gate $(sha256sum $X | cut -c1-16)" >> $S
for b in 1 2 4; do
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  d=$R/b$b; mkdir -p $d
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75
    DSV4_ROWS_GATE_PROFILE=$b nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt -o $d/nsys -f true $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64 9>&- > $d/gate.log 2>&1 )
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/nsys.nsys-rep > /dev/null 2>&1
  nsys stats --report cuda_api_sum --format csv -o $d/api $d/nsys.nsys-rep > /dev/null 2>&1
  rm -f $d/nsys.sqlite
  echo "rowsprof b=$b $(grep -h '^PROFILE' $d/gate.log)" >> $S
done
echo V3I_DONE >> $S
