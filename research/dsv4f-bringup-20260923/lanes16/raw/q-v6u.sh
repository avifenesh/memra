#!/usr/bin/env bash
# q-v6u.sh (SE pair): the lanes lane rebased on main $2 with its 16-lane default ($1), naked
# boots. cells-pdl M L L M (c1, c2, c4, 2k c2) against main, then cells-c24 on L, and the boot
# line that names the lane count.
set -u
LSHA=$1; MSHA=$2
S=/root/rcpt/q-v6u.summary; R=/root/rcpt/lanes16-default-v6u; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 16 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin main lane/dsv4-lanes8-20260927
git -C /root/lane/t-lanes checkout -q --detach $LSHA; git -C /root/lane/t-main13 checkout -q --detach $MSHA
[[ "$(git -C /root/lane/t-lanes rev-parse HEAD)" == "$LSHA" && "$(git -C /root/lane/t-main13 rev-parse HEAD)" == "$MSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-main13 /root/lane/target-main13 main15
mkdir -p /root/lane/bin-M15; cp /root/lane/target-main13/release/memra-server /root/lane/bin-M15/
bash /root/box/build.sh /root/lane/t-lanes /root/lane/target-tile lanes4
mkdir -p /root/lane/bin-L4; cp /root/lane/target-tile/release/memra-server /root/lane/bin-L4/
echo "build M $(grep -hE 'EXIT' /root/build-main15.log | tr '\n' ' ') L $(grep -hE 'EXIT' /root/build-lanes4.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-M15/memra-server /root/lane/bin-L4/memra-server > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
bin() { case $1 in M) echo /root/lane/bin-M15 ;; *) echo /root/lane/bin-L4 ;; esac; }
i=0
for a in M L L M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hoE 'serving lane\(s\)[^)]*\)' $d/serve.log | head -1) $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
d=$R/c24-L; wait_lock
timeout -k 30 3600 /root/box/cell.sh $d /root/lane/bin-L4/memra-server /root/box/cells-c24.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
echo "serve c24 L rc=$? $(grep -hoE '[0-9]+ serving lane\(s\)[^)]*\)' $d/serve.log | head -1) $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
echo "V6U_DONE $LSHA" >> $S
