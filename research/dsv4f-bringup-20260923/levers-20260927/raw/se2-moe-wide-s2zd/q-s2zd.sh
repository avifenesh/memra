#!/usr/bin/env bash
# q-s2zd.sh (second SE pair): the fused MoE pair at 16-warp CTAs ($1) against main (bin-M3,
# c566d2096). Component tests, long gate hash, TP/EP rows, DSpark TP/EP, the long gate
# M W W M M W and served cells-pdl M W W M.
set -u
WSHA=$1
S=/root/rcpt/q-s2zd.summary; R=/root/rcpt/moe-wide-s2zd; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 16 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-wide-cta-20260927
git worktree add -f /root/lane/t-wide $WSHA > /dev/null 2>&1; git -C /root/lane/t-wide checkout -q --detach $WSHA
[[ "$(git -C /root/lane/t-wide rev-parse HEAD)" == "$WSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-wide /root/lane/target-hm wide
X=/root/lane/target-hm/release; mkdir -p /root/lane/bin-W2; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-W2/
echo "build W $(git -C /root/lane/t-wide rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-wide.log | tr '\n' ' ')" >> $S
( cd /root/lane/t-wide && CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
sha256sum /root/lane/bin-M3/* /root/lane/bin-W2/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-wide && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_fused cuda_mrow_stream_matches cuda_deferred_partition cuda_tp_ep_local_only )
echo "component rc=$? $(grep -hE '^test result|FAILED|panicked' $R/component/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
XW=/root/lane/bin-W2
locked $R/long-304 $XW/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 W rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XW/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XW/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep W $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-M3 ;; *) echo /root/lane/bin-W2 ;; esac; }
i=0
for a in M W W M M W; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M W W M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2ZD_DONE $WSHA" >> $S
