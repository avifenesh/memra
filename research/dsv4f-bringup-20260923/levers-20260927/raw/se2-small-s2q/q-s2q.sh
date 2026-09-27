#!/usr/bin/env bash
# q-s2q.sh (second SE pair): the small-kernel lane $1 stacked on the dense pair lane $2, both
# rebased on main. Gates on the stack (long gate hash, TP/EP rows gate, DSpark TP/EP), then the
# long gate's replay ms/token P S S P and served P S S P P S (N=3), P = the rebased pair lane.
set -u
SSHA=$1; PSHA=$2
S=/root/rcpt/q-s2q.summary; R=/root/rcpt/small-regs-s2q; mkdir -p $R
until grep -q S2P_DONE /root/rcpt/q-s2p.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-small-regs-20260927 lane/dsv4-dense-pair-20260927
git -C /root/lane/t-pair checkout -q --detach $PSHA
bash /root/box/build.sh /root/lane/t-pair /root/lane/target-pair pair2
echo "build pair2 $(git -C /root/lane/t-pair rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-pair2.log | tr '\n' ' ')" >> $S
git worktree add -f /root/lane/t-small $SSHA > /dev/null 2>&1; git -C /root/lane/t-small checkout -q --detach $SSHA
[[ -d /root/lane/target-small ]] || cp -a /root/lane/target-pair /root/lane/target-small
bash /root/box/build.sh /root/lane/t-small /root/lane/target-small small
echo "build small $(git -C /root/lane/t-small rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-small.log | tr '\n' ' ')" >> $S
[[ "$(git -C /root/lane/t-small rev-parse HEAD)" == "$SSHA" && "$(git -C /root/lane/t-pair rev-parse HEAD)" == "$PSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
XF=/root/lane/target-mft/release; XP=/root/lane/target-pair/release; XS=/root/lane/target-small/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
sha256sum $XS/memra-server $XS/dsv4_tp_replay_long_gate $XS/dsv4_rows_gate $XS/dsv4-gpu-dspark-gate > $R/binaries.sha256
locked $R/long-304 $XS/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 S rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XS/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XS/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep S $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
bin() { case $1 in F) echo $XF ;; P) echo $XP ;; *) echo $XS ;; esac; }
i=0
for a in P S S P; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in P S S P P S; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2Q_DONE $SSHA" >> $S
