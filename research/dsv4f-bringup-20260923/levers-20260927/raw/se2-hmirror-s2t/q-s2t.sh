#!/usr/bin/env bash
# q-s2t.sh (second SE pair): the published h mirror lane $1 against main (target-small2, the
# engine tree of origin/main). Component tests, gates (long gate hash, TP/EP rows, DSpark TP/EP),
# then the long gate's replay ms/token M H H M M H and served M H H M M H (cells-pdl).
set -u
HSHA=$1
S=/root/rcpt/q-s2t.summary; R=/root/rcpt/hmirror-s2t; mkdir -p $R
rm -rf /root/lane/target-mft /root/lane/bin-mftG /root/lane/bin-mftH
git -C /root/lane/memra worktree remove --force /root/lane/t-probe > /dev/null 2>&1
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main lane/dsv4-moe-hmirror-20260927
git worktree add -f /root/lane/t-hm $HSHA > /dev/null 2>&1; git -C /root/lane/t-hm checkout -q --detach $HSHA
[[ "$(git -C /root/lane/t-hm rev-parse HEAD)" == "$HSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
[[ -d /root/lane/target-hm ]] || cp -a /root/lane/target-small2 /root/lane/target-hm
bash /root/box/build.sh /root/lane/t-hm /root/lane/target-hm hm
echo "build hm $(git -C /root/lane/t-hm rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-hm.log | tr '\n' ' ')" >> $S
( cd /root/lane/t-hm && CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
echo "test-build rc=$? $(tail -1 $R/test-build.log | cut -c1-120)" >> $S
XM=/root/lane/target-small2/release; XH=/root/lane/target-hm/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
sha256sum $XH/memra-server $XH/dsv4_tp_replay_long_gate $XH/dsv4_rows_gate $XH/dsv4-gpu-dspark-gate $XM/memra-server $XM/dsv4_tp_replay_long_gate > $R/binaries.sha256
( cd /root/lane/t-hm && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_fused cuda_mrow_stream_matches cuda_deferred_partition cuda_tp_ep_local_only )
echo "component rc=$? $(grep -hE '^test result|FAILED|panicked' $R/component/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
locked $R/long-304 $XH/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 H rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XH/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XH/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep H $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo $XM ;; *) echo $XH ;; esac; }
i=0
for a in M H H M M H; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M H H M M H; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2T_DONE $HSHA" >> $S
