#!/usr/bin/env bash
# q-s2zj.sh (second SE pair): the B-row diet L1..L5 ($1: compressor hoist at any width,
# grouped M-row wo_a, multi-row replay compressors, index lists, indexer, gather and attention)
# against main 80f734c77 (bin-ziM from q-s2zi).
# Long gate hash, TP/EP rows, KV split, DSpark TP/EP, rows wide 8, the rows gate's wide-16 timing
# M L L M, served cells-c24 M L L M and cells-pdl M L.
set -u
LSHA=$1
S=/root/rcpt/q-s2zj.summary; R=/root/rcpt/brow-diet-s2zj; mkdir -p $R
until grep -q S2ZI_DONE /root/rcpt/q-s2zi.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
X=/root/lane/target-hm/release
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-diet-20260927
git worktree add -f /root/lane/t-zj $LSHA > /dev/null 2>&1; git -C /root/lane/t-zj checkout -q --detach $LSHA
[[ "$(git -C /root/lane/t-zj rev-parse HEAD)" == "$LSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-zj /root/lane/target-hm zj
mkdir -p /root/lane/bin-zjL; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-zjL/
echo "build L $(git -C /root/lane/t-zj rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-zj.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-ziM/* /root/lane/bin-zjL/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
XL=/root/lane/bin-zjL
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
bin() { case $1 in M) echo /root/lane/bin-ziM ;; *) echo /root/lane/bin-zjL ;; esac; }
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
git -C /root/lane/memra worktree remove --force /root/lane/t-zj; git -C /root/lane/memra worktree prune
echo "S2ZJ_DONE $LSHA" >> $S
