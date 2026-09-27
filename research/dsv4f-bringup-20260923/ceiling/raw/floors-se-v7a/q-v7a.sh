#!/usr/bin/env bash
# q-v7a.sh (SE pair): the 1024-thread greedy argmax ($1) against main 80f734c77 (bin-M16).
# Component test, long gate hash, TP/EP rows, DSpark TP/EP, the launch floor (PDL on and off),
# the replayed step's DRAM traffic from nsys GPU metrics,
# the long gate M A A M M A, served cells-pdl M A A M.
set -u
ASHA=$1
S=/root/rcpt/q-v7a.summary; R=/root/rcpt/argmax-v7a; mkdir -p $R
until grep -q V6Z_DONE /root/rcpt/q-v6z.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
X=/root/lane/target-main13/release
cd /root/lane/memra && git fetch -q origin lane/dsv4-argmax-20260927
git worktree add -f /root/lane/t-am $ASHA > /dev/null 2>&1; git -C /root/lane/t-am checkout -q --detach $ASHA
[[ "$(git -C /root/lane/t-am rev-parse HEAD)" == "$ASHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-am /root/lane/target-main13 am
mkdir -p /root/lane/bin-AM; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-AM/
echo "build A $(git -C /root/lane/t-am rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-am.log | tr '\n' ' ')" >> $S
( cd /root/lane/t-am && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
sha256sum /root/lane/bin-M16/memra-server /root/lane/bin-M16/dsv4_tp_replay_long_gate /root/lane/bin-AM/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-am && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_argmax_wide )
echo "component rc=$? $(grep -hE '^test result|FAILED|panicked|^PASS' $R/component/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
XA=/root/lane/bin-AM
locked $R/long-304 $XA/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 A rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XA/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XA/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep A $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
locked $R/floor env DSV4_REPLAY_GATE_FLOOR=1 $XA/dsv4_tp_replay_long_gate $M $T 304
echo "floor pdl1 rc=$? $(grep -hE '^FLOOR|panicked' $R/floor/gate.log | sed 's/kernels_emptied_kept=//' | cut -c1-110 | tr '\n' ' ')" >> $S
locked $R/floor-pdl0 env MEMRA_DSV4_PDL=0 DSV4_REPLAY_GATE_FLOOR=1 $XA/dsv4_tp_replay_long_gate $M $T 304
echo "floor pdl0 rc=$? $(grep -hE '^FLOOR|panicked' $R/floor-pdl0/gate.log | sed 's/kernels_emptied_kept=//' | cut -c1-110 | tr '\n' ' ')" >> $S
d=$R/gm; mkdir -p $d; wait_lock
( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env DSV4_REPLAY_GATE_PROFILE=replay NVIDIA_TF32_OVERRIDE=0 timeout 2400 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --gpu-metrics-devices=all --gpu-metrics-set=gb20x --gpu-metrics-frequency=20000 -t cuda --cuda-graph-trace=graph -f true -o $d/gm $XA/dsv4_tp_replay_long_gate $M $T 64 9>&- > $d/gate.log 2>&1 )
nsys export --type sqlite -f true -o $d/gm.sqlite $d/gm.nsys-rep > $d/export.log 2>&1
python3 /root/box/gm_parse.py $d/gm.sqlite 64 > $d/gm.txt 2>&1
echo "gpu-metrics rc=$? $(grep -h 'PROFILE' $d/gate.log | tr '\n' ' ') $(grep -hE 'DRAM Read' $d/gm.txt | cut -c1-140 | tr '\n' ' ')" >> $S
rm -f $d/gm.sqlite
bin() { case $1 in M) echo /root/lane/bin-M16 ;; *) echo /root/lane/bin-AM ;; esac; }
i=0
for a in M A A M M A; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M A A M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
git -C /root/lane/memra worktree remove --force /root/lane/t-am; git -C /root/lane/memra worktree prune
echo "V7A_DONE $ASHA" >> $S
