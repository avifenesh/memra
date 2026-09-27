#!/usr/bin/env bash
# q-s2h.sh (second SE pair): the round lane rebased on main (push joins merged), $1.
#   1. Build main ($2) and the rebased lane.
#   2. Gates on the rebased lane: long-304 program hash, position-split store, DSpark TP/EP
#      proposal shas (must equal the pre-rebase E and C shas on this pod).
#   3. Served DSpark confirmation on the new base, order M R R M R M (N=3), greedy and sampled c1.
set -u
DSHA=$1; MSHA=$2
S=/root/rcpt/q-s2h.summary; R=/root/rcpt/dspark-round-s2h; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main lane/dsv4-dspark-round-20260926
git -C /root/lane/t-dsr checkout -q --detach $DSHA
git worktree add -f /root/lane/t-main $MSHA > /dev/null 2>&1; git -C /root/lane/t-main checkout -q --detach $MSHA
[[ -d /root/lane/target-main ]] || cp -a /root/lane/target-push /root/lane/target-main
bash /root/box/build.sh /root/lane/t-main /root/lane/target-main main
echo "build main $(git -C /root/lane/t-main rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-main.log | tr '\n' ' ') server $(sha256sum /root/lane/target-main/release/memra-server | cut -c1-16)" >> $S
bash /root/box/build.sh /root/lane/t-dsr /root/lane/target-dsr dsr2
echo "build dsr2 $(git -C /root/lane/t-dsr rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-dsr2.log | tr '\n' ' ') server $(sha256sum /root/lane/target-dsr/release/memra-server | cut -c1-16)" >> $S
XM=/root/lane/target-main/release; XR=/root/lane/target-dsr/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
sha256sum $XR/memra-server $XR/dsv4_tp_replay_long_gate $XR/dsv4_kv_split_gate $XR/dsv4-gpu-dspark-gate $XM/memra-server > $R/binaries.sha256
locked $R/long-304-R $XR/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 R rc=$? $(grep -hE 'PASS:|FAILED|FIRST|PROGRAM_SHA256|panicked' $R/long-304-R/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
locked $R/split-R $XR/dsv4_kv_split_gate $M $T 3000 300
echo "split R rc=$? $(grep -hE 'PASS: position|FAILED|panicked' $R/split-R/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
d=$R/dspark-tpep-R; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XR/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep R $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'verdict|GATE \[|GATE_DONE' $d/gate.log | head -n 6 | cut -c1-160 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo $XM ;; *) echo $XR ;; esac; }
i=0
for arm in M R R M R M; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $arm)/memra-server /root/box/cells-dsp.txt MEMRA_ENV_AUDIT=on MEMRA_DSV4_DRAFTER=dspark > $d.out 2>&1
  echo "serve r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2H_DONE $DSHA" >> $S
