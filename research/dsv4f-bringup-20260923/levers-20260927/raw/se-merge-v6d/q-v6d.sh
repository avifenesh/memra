#!/usr/bin/env bash
# q-v6d.sh (SE pair): the fused TP/EP MoE merge on this pair: main before it ($2, df006602e)
# against main with it ($1), one boot per row A B B A A B (N=3), naked defaults. The second pair's
# c4 rows were bimodal in both arms; this pair's c4 has been stable.
set -u
NSHA=$1
S=/root/rcpt/q-v6d.summary; R=/root/rcpt/fused-merge-v6d; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin main
git worktree add -f /root/lane/t-main13 $NSHA > /dev/null 2>&1; git -C /root/lane/t-main13 checkout -q --detach $NSHA
[[ -d /root/lane/target-main13 ]] || cp -a /root/lane/target-main12 /root/lane/target-main13
bash /root/box/build.sh /root/lane/t-main13 /root/lane/target-main13 main13
echo "build main13 $(git -C /root/lane/t-main13 rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-main13.log | tr '\n' ' ')" >> $S
XA=/root/lane/target-main12/release; XB=/root/lane/target-main13/release
sha256sum $XA/memra-server $XB/memra-server > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
bin() { case $1 in A) echo $XA ;; *) echo $XB ;; esac; }
i=0
for a in A B B A A B; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "V6D_DONE $NSHA" >> $S
