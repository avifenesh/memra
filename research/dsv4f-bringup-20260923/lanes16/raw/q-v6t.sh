#!/usr/bin/env bash
# q-v6t.sh (SE pair): the one-workspace coalescer fix ($1) against the build before it (bin-L2).
# Served cells-c24, one boot per row: O (bin-L2, 4 lanes), N (the fix, 4 lanes), W (the fix, 16
# lanes and rows), order O N W W N O.
set -u
LSHA=$1
S=/root/rcpt/q-v6t.summary; R=/root/rcpt/coalesce-v6t; mkdir -p $R
until grep -q V6S_DONE /root/rcpt/q-v6s.summary 2>/dev/null; do sleep 60; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-lanes8-20260927
git -C /root/lane/t-lanes checkout -q --detach $LSHA
[[ "$(git -C /root/lane/t-lanes rev-parse HEAD)" == "$LSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-lanes /root/lane/target-tile lanes3
X=/root/lane/target-tile/release; mkdir -p /root/lane/bin-L3; cp $X/memra-server /root/lane/bin-L3/
echo "build L3 $(git -C /root/lane/t-lanes rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-lanes3.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-L2/memra-server /root/lane/bin-L3/memra-server > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
arm() { case $1 in O) echo "/root/lane/bin-L2/memra-server MEMRA_ENV_AUDIT=on" ;; N) echo "/root/lane/bin-L3/memra-server MEMRA_ENV_AUDIT=on" ;; W) echo "/root/lane/bin-L3/memra-server MEMRA_ENV_AUDIT=on MEMRA_DSV4_SESSIONS=16 MEMRA_DSV4_ROWS=16" ;; esac; }
i=0
for a in O N W W N O; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  set -- $(arm $a); bin=$1; shift
  timeout -k 30 3600 /root/box/cell.sh $d $bin /root/box/cells-c24.txt "$@" > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "V6T_DONE $LSHA" >> $S
