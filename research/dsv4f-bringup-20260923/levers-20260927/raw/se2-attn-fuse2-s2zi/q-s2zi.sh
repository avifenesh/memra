#!/usr/bin/env bash
# q-s2zi.sh (second SE pair): the indexer q chain and o's rope-inverse pack ($1) against main ($2).
# Long gate hash, TP/EP rows, KV split, DSpark TP/EP, the long gate M F F M M F, served cells-pdl
# M F F M; then the lane's worktrees are removed.
set -u
FSHA=$1; MSHA=$2
S=/root/rcpt/q-s2zi.summary; R=/root/rcpt/attn-fuse2-s2zi; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main lane/dsv4-attn-small-fuse2-20260927
X=/root/lane/target-hm/release
for v in M:$MSHA F:$FSHA; do IFS=: read n sha <<< "$v"
  git worktree add -f /root/lane/t-zi$n $sha > /dev/null 2>&1; git -C /root/lane/t-zi$n checkout -q --detach $sha
  [[ "$(git -C /root/lane/t-zi$n rev-parse HEAD)" == "$sha" ]] || { echo "TREE_MISMATCH $n" >> $S; exit 1; }
  bash /root/box/build.sh /root/lane/t-zi$n /root/lane/target-hm zi$n
  mkdir -p /root/lane/bin-zi$n; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-zi$n/
  echo "build $n $(git -C /root/lane/t-zi$n rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-zi$n.log | tr '\n' ' ')" >> $S
done
sha256sum /root/lane/bin-zi*/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
XF=/root/lane/bin-ziF
locked $R/long-304 $XF/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 F rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XF/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
locked $R/kv-split env DSV4_KV_SPLIT_GATE_MAX_SEQ=1048576 $XF/dsv4_kv_split_gate $M $T 3000 100
echo "kv-split rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/kv-split/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XF/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep F $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-ziM ;; *) echo /root/lane/bin-ziF ;; esac; }
i=0
for a in M F F M M F; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M F F M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
for n in M F; do git -C /root/lane/memra worktree remove --force /root/lane/t-zi$n; done; git -C /root/lane/memra worktree prune
echo "S2ZI_DONE $FSHA" >> $S
