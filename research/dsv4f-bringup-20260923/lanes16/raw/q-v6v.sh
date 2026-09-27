#!/usr/bin/env bash
# q-v6v.sh (SE pair): the lanes lane with the contribution clear sized to the step ($1). Long gate
# hash, TP/EP rows gate with the 16-row wide phase, then naked cells-pdl M L L M against main
# (bin-M15) and one naked cells-c24 row.
set -u
LSHA=$1
S=/root/rcpt/q-v6v.summary; R=/root/rcpt/lanes16-clear-v6v; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-lanes8-20260927
git -C /root/lane/t-lanes checkout -q --detach $LSHA
[[ "$(git -C /root/lane/t-lanes rev-parse HEAD)" == "$LSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-lanes /root/lane/target-tile lanes5
X=/root/lane/target-tile/release; mkdir -p /root/lane/bin-L5; cp $X/memra-server $X/dsv4_rows_gate $X/dsv4_tp_replay_long_gate /root/lane/bin-L5/
echo "build L5 $(git -C /root/lane/t-lanes rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-lanes5.log | tr '\n' ' ')" >> $S
sha256sum /root/lane/bin-M15/memra-server /root/lane/bin-L5/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 5400 "$@" 9>&- > $d/gate.log 2>&1 ); }
locked $R/long-304 /root/lane/bin-L5/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 L5 rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-wide16 env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=16 /root/lane/bin-L5/dsv4_rows_gate $M $T 24 64
echo "rows wide16 rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-wide16/gate.log | cut -c1-90 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-M15 ;; *) echo /root/lane/bin-L5 ;; esac; }
i=0
for a in M L L M; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
d=$R/c24-L; wait_lock
timeout -k 30 3600 /root/box/cell.sh $d /root/lane/bin-L5/memra-server /root/box/cells-c24.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
echo "serve c24 L rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
echo "V6V_DONE $LSHA" >> $S
