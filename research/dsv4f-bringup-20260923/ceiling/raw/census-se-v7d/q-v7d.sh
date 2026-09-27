#!/usr/bin/env bash
# q-v7d.sh (SE pair): the step census on main with dense-fast to 16 rows ($1): the B-row step at 16
# and 4 rows (rows gate profile mode, captured TP/EP steps) and the one-row replayed step (long gate
# profile mode), nsys graph-node tracing, PDL off.
set -u
DSHA=$1
S=/root/rcpt/q-v7d.summary; R=/root/rcpt/census-v7d; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
X=/root/lane/target-main13/release
cd /root/lane/memra && git fetch -q origin lane/dsv4-dense16-20260927
git worktree add -f /root/lane/t-7d $DSHA > /dev/null 2>&1; git -C /root/lane/t-7d checkout -q --detach $DSHA
[[ "$(git -C /root/lane/t-7d rev-parse HEAD)" == "$DSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-7d /root/lane/target-main13 7d
mkdir -p /root/lane/bin-7d; cp $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate /root/lane/bin-7d/
echo "build $(git -C /root/lane/t-7d rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-7d.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-7d/* > $R/binaries.sha256
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
for b in 16 4; do
  d=$R/rows-b$b; mkdir -p $d
  ( NVIDIA_TF32_OVERRIDE=0 MEMRA_DSV4_PDL=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_PROFILE=$b DSV4_ROWS_GATE_PROFILE_GRAPH=1 timeout 3000 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --cuda-graph-trace=node -o $d/nsys -f true /root/lane/bin-7d/dsv4_rows_gate $M $T 24 64 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 64 > $d/ana.txt 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/nsys.nsys-rep > /dev/null 2>&1
  rm -f $d/nsys.sqlite $d/nsys.nsys-rep
  echo "census b$b $(grep -hE 'PROFILE|FAILED|panicked' $d/gate.log | cut -c1-120 | tr '\n' ' ') | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
d=$R/replay-b1; mkdir -p $d
( NVIDIA_TF32_OVERRIDE=0 MEMRA_DSV4_PDL=0 DSV4_REPLAY_GATE_PROFILE=replay timeout 3000 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --cuda-graph-trace=node -o $d/nsys -f true /root/lane/bin-7d/dsv4_tp_replay_long_gate $M $T 64 9>&- > $d/gate.log 2>&1 )
nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
python3 /root/box/nsys-ana.py $d/nsys.sqlite 64 > $d/ana.txt 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/nsys.nsys-rep > /dev/null 2>&1
rm -f $d/nsys.sqlite $d/nsys.nsys-rep
echo "census replay b1 $(grep -hE 'PROFILE|FAILED|panicked' $d/gate.log | cut -c1-120 | tr '\n' ' ') | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
git -C /root/lane/memra worktree remove --force /root/lane/t-7d; git -C /root/lane/memra worktree prune
echo "V7D_DONE $DSHA" >> $S
