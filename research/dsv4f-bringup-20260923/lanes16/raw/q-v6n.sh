#!/usr/bin/env bash
# q-v6n.sh (SE pair): serving lanes and B-row width up to 16 (memra #667), lane $1. Rows gate on
# TP/EP with the wide phase at 8 and at 16 rows, then served cells-c16 (c4, c8, c16 greedy,
# sampled c8), one boot per row: D (the default 4 lanes) E8 (8 lanes, 8 rows) E16 (16 and 16)
# in the order D E8 E16 E16 E8 D.
set -u
LSHA=$1
S=/root/rcpt/q-v6n.summary; R=/root/rcpt/lanes16-v6n; mkdir -p $R
until grep -q V6M_DONE /root/rcpt/q-v6m.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 16 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-lanes8-20260927
git worktree add -f /root/lane/t-lanes $LSHA > /dev/null 2>&1; git -C /root/lane/t-lanes checkout -q --detach $LSHA
[[ "$(git -C /root/lane/t-lanes rev-parse HEAD)" == "$LSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-lanes /root/lane/target-tile lanes
X=/root/lane/target-tile/release; mkdir -p /root/lane/bin-L; cp $X/memra-server $X/dsv4_rows_gate /root/lane/bin-L/
echo "build L $(git -C /root/lane/t-lanes rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-lanes.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-L/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 5400 "$@" 9>&- > $d/gate.log 2>&1 ); }
for w in 8 16; do
  locked $R/rows-wide$w env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=$w /root/lane/bin-L/dsv4_rows_gate $M $T 24 64
  echo "rows wide$w rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-wide$w/gate.log | cut -c1-110 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $R/rows-wide$w/gate.log | tr '\n' ' ')" >> $S
done
arm_env() { case $1 in D) echo "MEMRA_ENV_AUDIT=on" ;; E8) echo "MEMRA_ENV_AUDIT=on MEMRA_DSV4_SESSIONS=8 MEMRA_DSV4_ROWS=8" ;; E16) echo "MEMRA_ENV_AUDIT=on MEMRA_DSV4_SESSIONS=16 MEMRA_DSV4_ROWS=16" ;; esac; }
i=0
for a in D E8 E16 E16 E8 D; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d /root/lane/bin-L/memra-server /root/box/cells-c16.txt $(arm_env $a) > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "V6N_DONE $LSHA" >> $S
