#!/usr/bin/env bash
# q-v6z.sh (SE pair): the B-row diet L1+L2 ($1: compressor hoist at any width in 8-row chunks,
# grouped M-row wo_a) against main 80f734c77 (bin-M16, copied from q-v6y's build). Component
# tests, long gate hash, TP/EP rows (wide 16 and 8), KV split, DSpark TP/EP, the rows gate's
# wide timing M L L M, served cells-c24 M L L M and cells-pdl M L.
set -u
LSHA=$1
S=/root/rcpt/q-v6z.summary; R=/root/rcpt/brow-diet-v6z; mkdir -p $R
until grep -q V6Y_DONE /root/rcpt/q-v6y.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
X=/root/lane/target-main13/release
mkdir -p /root/lane/bin-M16; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-M16/
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-diet-20260927
git worktree add -f /root/lane/t-bd $LSHA > /dev/null 2>&1; git -C /root/lane/t-bd checkout -q --detach $LSHA
[[ "$(git -C /root/lane/t-bd rev-parse HEAD)" == "$LSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-bd /root/lane/target-main13 bd
mkdir -p /root/lane/bin-BD; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-BD/
echo "build L $(git -C /root/lane/t-bd rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-bd.log | tr '\n' ' ')" >> $S
( cd /root/lane/t-bd && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
sha256sum /root/lane/bin-M16/* /root/lane/bin-BD/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-bd && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_gemv_fp8_grouped cuda_dense_fast )
echo "component rc=$? $(grep -hE '^test result|FAILED|panicked|^PASS' $R/component/gate.log | tr '\n' ' ' | cut -c1-400)" >> $S
XL=/root/lane/bin-BD
locked $R/long-304 $XL/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 L rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XL/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
locked $R/rows-wide8 env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=8 $XL/dsv4_rows_gate $M $T 24 64
echo "rows wide8 rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-wide8/gate.log | cut -c1-120 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $R/rows-wide8/gate.log | tr '\n' ' ')" >> $S
locked $R/kv-split env DSV4_KV_SPLIT_GATE_MAX_SEQ=1048576 $XL/dsv4_kv_split_gate $M $T 3000 100
echo "kv-split rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/kv-split/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XL/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep L $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-M16 ;; *) echo /root/lane/bin-BD ;; esac; }
i=0
for a in M L L M; do
  i=$((i+1)); d=$R/wide16-t$i-$a
  locked $d env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=16 $(bin $a)/dsv4_rows_gate $M $T 24 64
  echo "wide16 t$i $a rc=$? $(grep -hE '^PASS|FAIL|panicked' $d/gate.log | cut -c1-80 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for spec in M:c24 L:c24 L:c24 M:c24 M:pdl L:pdl; do
  IFS=: read a c <<< "$spec"; i=$((i+1)); d=$R/r$i-$a-$c; wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-$c.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a $c rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
git -C /root/lane/memra worktree remove --force /root/lane/t-bd; git -C /root/lane/memra worktree prune
echo "V6Z_DONE $LSHA" >> $S
