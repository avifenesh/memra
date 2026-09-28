#!/usr/bin/env bash
# q-s2zm.sh (second SE pair): the owned-row expert join ($1) against the B-row lane it stacks on
# (bin-zlB from q-s2zl). Long gate hash, TP/EP rows, wide 16, KV split, the long gate B O O B B O,
# served cells-c24 B O O B and cells-pdl B O.
set -u
OSHA=$1
S=/root/rcpt/q-s2zm.summary; R=/root/rcpt/owned-join-s2zm; mkdir -p $R
until grep -q S2ZL_DONE /root/rcpt/q-s2zl.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
X=/root/lane/target-hm/release
cd /root/lane/memra && git fetch -q origin lane/dsv4-owned-join-20260928
git worktree add -f /root/lane/t-zm $OSHA > /dev/null 2>&1; git -C /root/lane/t-zm checkout -q --detach $OSHA
[[ "$(git -C /root/lane/t-zm rev-parse HEAD)" == "$OSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-zm /root/lane/target-hm zm
mkdir -p /root/lane/bin-zmO; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate /root/lane/bin-zmO/
echo "build O $(git -C /root/lane/t-zm rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-zm.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-zlB/memra-server /root/lane/bin-zlB/dsv4_tp_replay_long_gate /root/lane/bin-zmO/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
XO=/root/lane/bin-zmO
locked $R/long-304 $XO/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 O rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XO/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
locked $R/rows-wide16 env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=16 $XO/dsv4_rows_gate $M $T 24 64
echo "rows wide16 rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-wide16/gate.log | cut -c1-90 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $R/rows-wide16/gate.log | tr '\n' ' ')" >> $S
locked $R/kv-split env DSV4_KV_SPLIT_GATE_MAX_SEQ=1048576 $XO/dsv4_kv_split_gate $M $T 3000 100
echo "kv-split rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/kv-split/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
bin() { case $1 in B) echo /root/lane/bin-zlB ;; *) echo /root/lane/bin-zmO ;; esac; }
i=0
for a in B O O B B O; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for spec in B:c24 O:c24 O:c24 B:c24 B:pdl O:pdl; do
  IFS=: read a c <<< "$spec"; i=$((i+1)); d=$R/r$i-$a-$c; wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-$c.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a $c rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
git -C /root/lane/memra worktree remove --force /root/lane/t-zm; git -C /root/lane/memra worktree prune
echo "S2ZM_DONE $OSHA" >> $S
