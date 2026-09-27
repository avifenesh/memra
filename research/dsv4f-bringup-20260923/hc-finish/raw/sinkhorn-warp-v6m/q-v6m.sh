#!/usr/bin/env bash
# q-v6m.sh (SE pair): the HC finish's Sinkhorn on a fifth warp, lane $1 against main $2
# (t-main13/target-main13 from q-v6k). One card: the HC finish bit test and its timing test,
# M W W M. Two cards: long gate hash, replay ms/token M W W M M W, served cells-pdl M W W M.
set -u
WSHA=$1; MSHA=$2
S=/root/rcpt/q-v6m.summary; R=/root/rcpt/hc-warp-v6m; mkdir -p $R
until grep -q V6L_DONE /root/rcpt/q-v6l.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 20 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
[[ "$(git -C /root/lane/t-main13 rev-parse HEAD)" == "$MSHA" ]] || { echo "TREE_MISMATCH main" >> $S; exit 1; }
( cd /root/lane/t-main13 && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --test dsv4_hc_finish_gpu --no-run -j 56 > $R/test-build-M.log 2>&1 )
TM=$(grep -oE '/root/lane/target-main13/release/deps/dsv4_hc_finish_gpu-[0-9a-f]+' $R/test-build-M.log | head -1)
mkdir -p /root/lane/bin-M14 /root/lane/bin-W
cp $TM /root/lane/bin-M14/hc_test; cp /root/lane/target-main13/release/memra-server /root/lane/target-main13/release/dsv4_tp_replay_long_gate /root/lane/bin-M14/
cd /root/lane/memra && git fetch -q origin lane/dsv4-hc-sinkhorn-warp-20260927
git worktree add -f /root/lane/t-hcw $WSHA > /dev/null 2>&1; git -C /root/lane/t-hcw checkout -q --detach $WSHA
[[ "$(git -C /root/lane/t-hcw rev-parse HEAD)" == "$WSHA" ]] || { echo "TREE_MISMATCH lane" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-hcw /root/lane/target-main13 hcw
( cd /root/lane/t-hcw && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --test dsv4_hc_finish_gpu --no-run -j 56 > $R/test-build-W.log 2>&1 )
TW=$(grep -oE '/root/lane/target-main13/release/deps/dsv4_hc_finish_gpu-[0-9a-f]+' $R/test-build-W.log | head -1)
cp $TW /root/lane/bin-W/hc_test; cp /root/lane/target-main13/release/memra-server /root/lane/target-main13/release/dsv4_tp_replay_long_gate /root/lane/bin-W/
echo "build W $(git -C /root/lane/t-hcw rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-hcw.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-M14/* /root/lane/bin-W/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
locked $R/bits env CUDA_VISIBLE_DEVICES=0 /root/lane/bin-W/hc_test --ignored --nocapture --test-threads=1 dsv4_hc_finish_is_bit_identical_to_the_unfused_chain
echo "bits W rc=$? $(grep -hE '^test result|FAILED|panicked|DSV4_HC_FINISH' $R/bits/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
i=0
for a in M14 W W M14; do
  i=$((i+1)); d=$R/timing-t$i-$a
  locked $d env CUDA_VISIBLE_DEVICES=0 /root/lane/bin-$a/hc_test --ignored --nocapture --test-threads=1 dsv4_hc_finish_timing
  echo "timing t$i $a rc=$? $(grep -hoE 'rows=[0-9]+ rep=[0-9]+ .*fused_us=[0-9.]+' $d/gate.log | sed -E 's/chain_us=[0-9.]+ diet_us=[0-9.]+ //' | tr '\n' ' ')" >> $S
done
i=0
for a in M14 W W M14 M14 W; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d /root/lane/bin-$a/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | sort -u | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M14 W W M14; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d /root/lane/bin-$a/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "V6M_DONE $WSHA" >> $S
