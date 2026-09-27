#!/usr/bin/env bash
# q-pair16.sh (2x RTX PRO 6000 WS pod): memra #710 TP/EP B-row anatomy, lane
# lane/dsv4-tp-rows-20260925 5d21db2a3. dsv4_rows_gate profile mode on TP/EP at B=1, 2 and 4:
# nsys (cudaProfilerApi) over 64 B-row steps each, then the per-device kernel table.
while ! grep -q PAIR15_DONE /root/rcpt/q-pair15.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair16.summary; R=/root/rcpt/tp-rows-anat; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-20260925
git worktree add -f /root/lane/t-rows 5d21db2a3 > /dev/null 2>&1; git -C /root/lane/t-rows checkout -q --detach 5d21db2a3
[[ -d /root/lane/target-rows ]] || cp -a /root/lane/target-flip /root/lane/target-rows
( cd /root/lane/t-rows && CARGO_TARGET_DIR=/root/lane/target-rows cargo build --release -j 56 -p memra-engine --bin dsv4_rows_gate ) > $R/build.log 2>&1
X=/root/lane/target-rows/release/dsv4_rows_gate
echo "build rows $(git -C /root/lane/t-rows rev-parse --short HEAD) rc=$? gate $(sha256sum $X | cut -c1-16)" >> $S
exec 9>/tmp/memra-gpu.lock
while ! flock -n 9; do sleep 10; done
for b in 1 2 4; do
  d=$R/b$b; mkdir -p $d
  ( NVIDIA_TF32_OVERRIDE=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_PROFILE=$b nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt -o $d/nsys -f true $X /data/dsv4f/nvfp4 /root/box/tape-rebuild.txt 24 64 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 64 > $d/ana.txt 2>&1
  rm -f $d/nsys.sqlite
  echo "anat B=$b $(grep -hE 'PROFILE|FAILED|panicked' $d/gate.log | cut -c1-160 | tr '\n' ' ') | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
echo PAIR16_DONE >> $S
