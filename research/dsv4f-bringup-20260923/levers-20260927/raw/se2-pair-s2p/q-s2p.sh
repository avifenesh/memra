#!/usr/bin/env bash
# q-s2p.sh (second SE pair): the dense pair lane $1 (stacked on the fused TP/EP MoE lane, whose
# build F is target-mft). Gates on the pair lane: the dense-fast component gate with its pair
# cases, the long gate's program hash, the TP/EP rows gate, the DSpark TP/EP gate. Then the long
# gate's replay ms/token F P P F and served one boot per row F P P F F P.
set -u
PSHA=$1
S=/root/rcpt/q-s2p.summary; R=/root/rcpt/dense-pair-s2p; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-dense-pair-20260927
git worktree add -f /root/lane/t-pair $PSHA > /dev/null 2>&1; git -C /root/lane/t-pair checkout -q --detach $PSHA
[[ -d /root/lane/target-pair ]] || cp -a /root/lane/target-mft /root/lane/target-pair
bash /root/box/build.sh /root/lane/t-pair /root/lane/target-pair pair
echo "build pair $(git -C /root/lane/t-pair rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-pair.log | tr '\n' ' ')" >> $S
XF=/root/lane/target-mft/release; XP=/root/lane/target-pair/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
mkdir -p /root/lane/cgate-pair
( cd /root/lane/t-pair && nice -n 10 nvcc -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off -arch=sm_120a -lineinfo tools/dsv4-dense-fast-gate.cu -lcublasLt -lcublas -ldl -o /root/lane/cgate-pair/component > $R/cgate-build.log 2>&1 )
locked $R/component /root/lane/cgate-pair/component --check-only
echo "component rc=$? $(grep -hE 'PASS pair|PASS dense_fast|FAIL' $R/component/gate.log | tail -3 | cut -c1-200 | tr '\n' ' ') pairs=$(grep -c 'PASS pair' $R/component/gate.log)" >> $S
sha256sum $XF/memra-server $XP/memra-server $XP/dsv4_tp_replay_long_gate $XP/dsv4_rows_gate $XP/dsv4-gpu-dspark-gate > $R/binaries.sha256
locked $R/long-304 $XP/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 P rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $R/long-304/gate.log | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XP/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XP/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep P $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE|FUSED MOE|MROW' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
bin() { case $1 in F) echo $XF ;; *) echo $XP ;; esac; }
i=0
for a in F P P F; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in F P P F F P; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2P_DONE $PSHA" >> $S
