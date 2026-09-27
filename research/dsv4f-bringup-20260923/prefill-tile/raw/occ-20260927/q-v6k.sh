#!/usr/bin/env bash
# q-v6k.sh (SE pair): the prefill tile lane $1 rebased on main $2. Gates on the lane (the tile bit
# test, long gate hash, TP/EP rows, KV split, DSpark TP/EP), then served TTFT M T T M M T (N=3),
# cells-ttft (8k and 32k prompts, c1 greedy, each with its decode rate).
set -u
TSHA=$1; MSHA=$2
S=/root/rcpt/q-v6k.summary; R=/root/rcpt/tile-rebased-v6k; mkdir -p $R
rm -rf /root/lane/target-mft /root/lane/target-push /root/lane/target-main12 /root/lane/bin-dots* /root/lane/bin-tile* /root/lane/bin-tsrv*
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 30 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main lane/dsv4-prefill-tile-occ-20260927
git -C /root/lane/t-main13 checkout -q --detach $MSHA
git -C /root/lane/t-tile checkout -q --detach $TSHA
[[ "$(git -C /root/lane/t-tile rev-parse HEAD)" == "$TSHA" && "$(git -C /root/lane/t-main13 rev-parse HEAD)" == "$MSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-main13 /root/lane/target-main13 main14
echo "build main $(git -C /root/lane/t-main13 rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-main14.log | tr '\n' ' ')" >> $S
bash /root/box/build.sh /root/lane/t-tile /root/lane/target-tile tile2
echo "build tile $(git -C /root/lane/t-tile rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-tile2.log | tr '\n' ' ')" >> $S
( cd /root/lane/t-tile && CARGO_TARGET_DIR=/root/lane/target-tile cargo test --release -p memra-engine --test dsv4_gemm_tile_gpu --no-run -j 56 > $R/test-build.log 2>&1 )
TB=$(grep -oE '/root/lane/target-tile/release/deps/dsv4_gemm_tile_gpu-[0-9a-f]+' $R/test-build.log | head -1)
echo "test-build $TB" >> $S
XM=/root/lane/target-main13/release; XT=/root/lane/target-tile/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
sha256sum $XT/memra-server $XT/dsv4_tp_replay_long_gate $XT/dsv4_rows_gate $XT/dsv4_kv_split_gate $XT/dsv4-gpu-dspark-gate $XM/memra-server $TB > $R/binaries.sha256
locked $R/tile-test env CUDA_VISIBLE_DEVICES=0 $TB --nocapture
echo "tile-test rc=$? $(grep -hE '^test result|FAILED|panicked' $R/tile-test/gate.log | tr '\n' ' ' | cut -c1-200)" >> $S
locked $R/long-304 $XT/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 T rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XT/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
locked $R/kv-split env DSV4_KV_SPLIT_GATE_MAX_SEQ=1048576 $XT/dsv4_kv_split_gate $M $T 3000 100
echo "kv-split rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/kv-split/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XT/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep T $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo $XM ;; *) echo $XT ;; esac; }
i=0
for a in M T T M M T; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-ttft.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "V6K_DONE $TSHA" >> $S
