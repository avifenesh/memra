#!/usr/bin/env bash
# q-v6y.sh (SE pair): the day's close. B = main 359e850d0 (bin-M14, before the Sinkhorn warp, the
# tile, the owner, lanes, tail, attention fusions and the router mirror); F = final main ($1).
# Long gate hash on F, the long gate B F F B, served cells-pdl B F F B, cells-c24 B F, cells-ttft B F.
set -u
FSHA=$1
S=/root/rcpt/q-v6y.summary; R=/root/rcpt/close-v6y; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin main
git worktree add -f /root/lane/t-final $FSHA > /dev/null 2>&1; git -C /root/lane/t-final checkout -q --detach $FSHA
[[ "$(git -C /root/lane/t-final rev-parse HEAD)" == "$FSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-final /root/lane/target-main13 final
X=/root/lane/target-main13/release; mkdir -p /root/lane/bin-F; cp $X/memra-server $X/dsv4_tp_replay_long_gate /root/lane/bin-F/
echo "build F $(git -C /root/lane/t-final rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-final.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-M14/memra-server /root/lane/bin-M14/dsv4_tp_replay_long_gate /root/lane/bin-F/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
bin() { case $1 in B) echo /root/lane/bin-M14 ;; *) echo /root/lane/bin-F ;; esac; }
i=0
for a in B F F B; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304' $d/gate.log | sort -u | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for spec in B:pdl F:pdl F:pdl B:pdl B:c24 F:c24 B:ttft F:ttft; do
  IFS=: read a c <<< "$spec"; i=$((i+1)); d=$R/r$i-$a-$c; wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-$c.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a $c rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-160 | tr '\n' ' ')" >> $S
done
git -C /root/lane/memra worktree remove --force /root/lane/t-final > /dev/null 2>&1; git -C /root/lane/memra worktree prune
echo "V6Y_DONE $FSHA" >> $S
