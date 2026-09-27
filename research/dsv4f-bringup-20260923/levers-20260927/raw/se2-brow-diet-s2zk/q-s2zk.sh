#!/usr/bin/env bash
# q-s2zk.sh (second SE pair): the B-row diet with captured steps up to 16 rows ($1) against main
# 80f734c77 (bin-ziM). Rows wide 16 (the first 16-row graph steps), rows TP/EP, wide 8, the long
# gate hash, the rows gate's wide-16 timing M L L M, served cells-c24 M L L M and cells-pdl M L.
set -u
LSHA=$1
S=/root/rcpt/q-s2zk.summary; R=/root/rcpt/brow-diet-s2zk; mkdir -p $R
until grep -q S2ZJ_DONE /root/rcpt/q-s2zj.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
X=/root/lane/target-hm/release
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-diet-20260927
git worktree add -f /root/lane/t-zk $LSHA > /dev/null 2>&1; git -C /root/lane/t-zk checkout -q --detach $LSHA
[[ "$(git -C /root/lane/t-zk rev-parse HEAD)" == "$LSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-zk /root/lane/target-hm zk
mkdir -p /root/lane/bin-zkL; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate /root/lane/bin-zkL/
echo "build L $(git -C /root/lane/t-zk rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-zk.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-ziM/memra-server /root/lane/bin-ziM/dsv4_rows_gate /root/lane/bin-zkL/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
XL=/root/lane/bin-zkL
bin() { case $1 in M) echo /root/lane/bin-ziM ;; *) echo /root/lane/bin-zkL ;; esac; }
i=0
for a in L M L M; do
  i=$((i+1)); d=$R/wide16-t$i-$a
  locked $d env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=16 $(bin $a)/dsv4_rows_gate $M $T 24 64
  echo "wide16 t$i $a rc=$? $(grep -hE '^PASS|FAIL|panicked|Error' $d/gate.log | cut -c1-90 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $d/gate.log | tr '\n' ' ')" >> $S
done
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XL/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
locked $R/rows-wide8 env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=8 $XL/dsv4_rows_gate $M $T 24 64
echo "rows wide8 rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-wide8/gate.log | cut -c1-90 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $R/rows-wide8/gate.log | tr '\n' ' ')" >> $S
locked $R/long-304 $XL/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 L rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
i=0
for spec in M:c24 L:c24 L:c24 M:c24 L:pdl M:pdl; do
  IFS=: read a c <<< "$spec"; i=$((i+1)); d=$R/r$i-$a-$c; wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-$c.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a $c rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
git -C /root/lane/memra worktree remove --force /root/lane/t-zk; git -C /root/lane/memra worktree prune
echo "S2ZK_DONE $LSHA" >> $S
