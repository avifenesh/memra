#!/usr/bin/env bash
# q-s2j.sh (second SE pair): the fused TP/EP MoE lane $1 against main $2.
#   1. Build; the fused component tests (partition and full bank) on card 0.
#   2. Gates on the lane: long gate program hash, TP/EP rows gate, position-split gate, DSpark
#      TP/EP (plain-arm fused engagement and proposal shas).
#   3. Long-gate replay timing M F F M, then served one boot per row M F F M M F F M M F (N=5).
set -u
FSHA=$1; MSHA=$2
S=/root/rcpt/q-s2j.summary; R=/root/rcpt/moe-fused-s2j; mkdir -p $R
until grep -q S2I_DONE /root/rcpt/q-s2i.summary 2>/dev/null; do sleep 60; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-fused-tpep-20260926
git worktree add -f /root/lane/t-mft $FSHA > /dev/null 2>&1; git -C /root/lane/t-mft checkout -q --detach $FSHA
[[ -d /root/lane/target-mft ]] || cp -a /root/lane/target-main /root/lane/target-mft
bash /root/box/build.sh /root/lane/t-mft /root/lane/target-mft mft
echo "build mft $(git -C /root/lane/t-mft rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-mft.log | tr '\n' ' ')" >> $S
XM=/root/lane/target-main/release; XF=/root/lane/target-mft/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
# component tests (the test harness builds into the same target dir)
( cd /root/lane/t-mft && CARGO_TARGET_DIR=/root/lane/target-mft cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
echo "test build rc=$? $(tail -1 $R/test-build.log | cut -c1-160)" >> $S
( cd /root/lane/t-mft && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-mft cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 cuda_fused cuda_m1_stream_matches cuda_deferred_partition cuda_tp_ep_local_only )
echo "component rc=$? $(grep -hE '^test |test result|EXACT fused partition|panicked' $R/component/gate.log | tail -14 | cut -c1-140 | tr '\n' ' ')" >> $S
sha256sum $XM/memra-server $XF/memra-server $XF/dsv4_tp_replay_long_gate $XF/dsv4_rows_gate $XF/dsv4_kv_split_gate $XF/dsv4-gpu-dspark-gate > $R/binaries.sha256
locked $R/long-304-F $XF/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 F rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked|ENGAGED on TP/EP[^,]*' $R/long-304-F/gate.log | sort -u | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $R/long-304-F/gate.log | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XF/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE 'PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-200 | tr '\n' ' ')" >> $S
locked $R/split $XF/dsv4_kv_split_gate $M $T 3000 300
echo "split rc=$? $(grep -hE 'PASS: position|FAILED|panicked' $R/split/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XF/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep F $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'arm P|verdict|GATE \[|GATE_DONE|FUSED MOE' $d/gate.log | head -n 8 | cut -c1-170 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo $XM ;; *) echo $XF ;; esac; }
i=0
for a in M F F M; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M F F M M F F M M F; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2J_DONE $FSHA" >> $S
