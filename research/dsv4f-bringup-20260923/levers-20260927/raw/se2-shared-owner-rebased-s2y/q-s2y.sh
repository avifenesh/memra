#!/usr/bin/env bash
# q-s2y.sh (second SE pair): the shared-expert owner lane rebased on main $2 (with the Sinkhorn
# warp and the prefill tile): $1. Builds main (M3) and the lane (S3) in one scratch target, then
# the component tests, the dense-fast gate's gated cases, long gate hash, TP/EP rows, KV split,
# DSpark TP/EP on S3, the long gate M3 S3 S3 M3 and served cells-pdl M3 S3 S3 M3.
set -u
SSHA=$1; MSHA=$2
S=/root/rcpt/q-s2y.summary; R=/root/rcpt/shared-owner-rebased-s2y; mkdir -p $R
until grep -q S2X_DONE /root/rcpt/q-s2x.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 16 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main lane/dsv4-shared-owner-main-20260927
git -C /root/lane/t-m2 checkout -q --detach $MSHA; git -C /root/lane/t-so checkout -q --detach $SSHA
[[ "$(git -C /root/lane/t-so rev-parse HEAD)" == "$SSHA" && "$(git -C /root/lane/t-m2 rev-parse HEAD)" == "$MSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
X=/root/lane/target-hm/release
bash /root/box/build.sh /root/lane/t-m2 /root/lane/target-hm m3
mkdir -p /root/lane/bin-M3; cp $X/memra-server $X/dsv4_tp_replay_long_gate /root/lane/bin-M3/
echo "build M3 $(git -C /root/lane/t-m2 rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-m3.log | tr '\n' ' ')" >> $S
bash /root/box/build.sh /root/lane/t-so /root/lane/target-hm s3
mkdir -p /root/lane/bin-S3; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-S3/
echo "build S3 $(git -C /root/lane/t-so rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-s3.log | tr '\n' ' ')" >> $S
( cd /root/lane/t-so && CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
mkdir -p $R/cgate; ( cd /root/lane/t-so && nvcc -t 8 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off -arch=sm_120a -lineinfo tools/dsv4-dense-fast-gate.cu -lcublasLt -lcublas -ldl -o $R/cgate/component > $R/cgate/build.log 2>&1 )
echo "cgate-build rc=$?" >> $S
sha256sum /root/lane/bin-M3/* /root/lane/bin-S3/* $R/cgate/component > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-so && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_fused cuda_mrow_stream_matches cuda_deferred_partition cuda_tp_ep_local_only )
echo "component rc=$? $(grep -hE '^test result|FAILED|panicked' $R/component/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
locked $R/cgate-run $R/cgate/component --check-only
echo "cgate rc=$? $(grep -hcE '^PASS gated' $R/cgate-run/gate.log) gated $(grep -hE 'PASS dense_fast_components|FAIL' $R/cgate-run/gate.log | tr '\n' ' ')" >> $S
XS=/root/lane/bin-S3
locked $R/long-304 $XS/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 S3 rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XS/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
locked $R/kv-split env DSV4_KV_SPLIT_GATE_MAX_SEQ=1048576 $XS/dsv4_kv_split_gate $M $T 3000 100
echo "kv-split rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/kv-split/gate.log | tail -3 | cut -c1-160 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XS/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep S3 $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-160 | tr '\n' ' ')" >> $S
i=0
for a in M3 S3 S3 M3; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d /root/lane/bin-$a/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M3 S3 S3 M3; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d /root/lane/bin-$a/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "S2Y_DONE $SSHA" >> $S
