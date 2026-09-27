#!/usr/bin/env bash
# q-s2zg.sh (second SE pair): the router-built x mirror ($1) against main ($2). Component tests,
# long gate hash, TP/EP rows, DSpark TP/EP, the long gate M X X M M X, served cells-pdl M X X M.
set -u
XSHA=$1; MSHA=$2
S=/root/rcpt/q-s2zg.summary; R=/root/rcpt/xmirror-s2zg; mkdir -p $R
until grep -q S2ZF_DONE /root/rcpt/q-s2zf.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 16 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main lane/dsv4-xmirror-route-20260927
X=/root/lane/target-hm/release
for v in M5:$MSHA X:$XSHA; do IFS=: read n sha <<< "$v"
  git worktree add -f /root/lane/t-zg$n $sha > /dev/null 2>&1; git -C /root/lane/t-zg$n checkout -q --detach $sha
  [[ "$(git -C /root/lane/t-zg$n rev-parse HEAD)" == "$sha" ]] || { echo "TREE_MISMATCH $n" >> $S; exit 1; }
  bash /root/box/build.sh /root/lane/t-zg$n /root/lane/target-hm zg$n
  mkdir -p /root/lane/bin-zg$n; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-zg$n/
  echo "build $n $(git -C /root/lane/t-zg$n rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-zg$n.log | tr '\n' ' ')" >> $S
done
( cd /root/lane/t-zgX && CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
sha256sum /root/lane/bin-zg*/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-zgX && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_fused cuda_mrow_stream_matches cuda_deferred_partition cuda_tp_ep_local_only )
echo "component rc=$? $(grep -hE '^test result|FAILED|panicked' $R/component/gate.log | tr '\n' ' ' | cut -c1-200)" >> $S
XB=/root/lane/bin-zgX
locked $R/long-304 $XB/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 X rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XB/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XB/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep X $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-zgM5 ;; *) echo /root/lane/bin-zgX ;; esac; }
i=0
for a in M X X M M X; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M X X M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2ZG_DONE $XSHA" >> $S
