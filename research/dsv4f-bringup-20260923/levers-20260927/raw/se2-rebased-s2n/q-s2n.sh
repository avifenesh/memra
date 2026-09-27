#!/usr/bin/env bash
# q-s2n.sh (second SE pair): the fused TP/EP MoE lane rebased on main ($1, the DSpark round lane
# and the route fix merged under it). Gates on the rebased tree: the component tests, the long
# gate's program hash, the TP/EP rows gate, the split gate and the DSpark TP/EP proposal shas.
set -u
FSHA=$1
S=/root/rcpt/q-s2n.summary; R=/root/rcpt/moe-fused-rebased-s2n; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 8 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-fused-tpep-20260926
git -C /root/lane/t-mft checkout -q -- . ; git -C /root/lane/t-mft checkout -q --detach $FSHA
bash /root/box/build.sh /root/lane/t-mft /root/lane/target-mft mft3
echo "build mft3 $(git -C /root/lane/t-mft rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-mft3.log | tr '\n' ' ')" >> $S
XF=/root/lane/target-mft/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-mft && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-mft cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_fused cuda_mrow_stream_matches cuda_deferred_partition cuda_tp_ep_local_only )
echo "component rc=$? $(grep -hE 'test result|EXACT fused partition rows|panicked' $R/component/gate.log | tail -4 | cut -c1-140 | tr '\n' ' ')" >> $S
sha256sum $XF/memra-server $XF/dsv4_tp_replay_long_gate $XF/dsv4_rows_gate $XF/dsv4_kv_split_gate $XF/dsv4-gpu-dspark-gate > $R/binaries.sha256
locked $R/long-304 $XF/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $R/long-304/gate.log | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XF/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
locked $R/split $XF/dsv4_kv_split_gate $M $T 3000 300
echo "split rc=$? $(grep -hE 'PASS: position|FAILED|panicked' $R/split/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XF/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'verdict|GATE \[|GATE_DONE|FUSED MOE' $d/gate.log | head -n 6 | cut -c1-160 | tr '\n' ' ')" >> $S
echo "S2N_DONE $FSHA" >> $S
