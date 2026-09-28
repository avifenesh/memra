#!/usr/bin/env bash
# q-v7e.sh (SE pair): four output rows per dense-fast block past 8 token rows ($1) against main
# with dense-fast to 16 rows (bin-7d, $2 is its sha for the record). Dense-fast rows component
# test, long gate hash, TP/EP rows, the rows gate's wide-16 timing R M R M, served cells-c24 M R R M.
set -u
RSHA=$1; MSHA=$2
S=/root/rcpt/q-v7e.summary; R=/root/rcpt/rows4-v7e; mkdir -p $R
until grep -q V7D_DONE /root/rcpt/q-v7d.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
X=/root/lane/target-main13/release
cd /root/lane/memra && git fetch -q origin lane/dsv4-dense16-rows4-20260928 main
# main's server for the served arm, built first (bin-7d holds only the gates)
git worktree add -f /root/lane/t-7eM $MSHA > /dev/null 2>&1; git -C /root/lane/t-7eM checkout -q --detach $MSHA
bash /root/box/build.sh /root/lane/t-7eM /root/lane/target-main13 7eM
mkdir -p /root/lane/bin-7eM; cp $X/memra-server $X/dsv4_rows_gate $X/dsv4_tp_replay_long_gate /root/lane/bin-7eM/
echo "build M $(git -C /root/lane/t-7eM rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-7eM.log | tr '\n' ' ')" >> $S
git worktree add -f /root/lane/t-7e $RSHA > /dev/null 2>&1; git -C /root/lane/t-7e checkout -q --detach $RSHA
[[ "$(git -C /root/lane/t-7e rev-parse HEAD)" == "$RSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
bash /root/box/build.sh /root/lane/t-7e /root/lane/target-main13 7e
mkdir -p /root/lane/bin-7eR; cp $X/memra-server $X/dsv4_rows_gate $X/dsv4_tp_replay_long_gate /root/lane/bin-7eR/
echo "build R $(git -C /root/lane/t-7e rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-7e.log | tr '\n' ' ')" >> $S
( cd /root/lane/t-7e && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --test dsv4_dense_fast_rows_gpu --no-run -j 56 > $R/test-build.log 2>&1 )
sha256sum /root/lane/bin-7eM/* /root/lane/bin-7eR/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-7e && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --test dsv4_dense_fast_rows_gpu -j 56 -- --ignored --test-threads=1 --nocapture )
echo "component rows rc=$? $(grep -hE '^test result|FAILED|panicked|cases' $R/component/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
XR=/root/lane/bin-7eR
locked $R/long-304 $XR/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 R rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XR/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
bin() { case $1 in M) echo /root/lane/bin-7eM ;; *) echo /root/lane/bin-7eR ;; esac; }
i=0
for a in R M R M; do
  i=$((i+1)); d=$R/wide16-t$i-$a
  locked $d env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=16 $(bin $a)/dsv4_rows_gate $M $T 24 64
  echo "wide16 t$i $a rc=$? $(grep -hE '^PASS|FAIL|panicked' $d/gate.log | cut -c1-60 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in M R R M; do
  i=$((i+1)); d=$R/r$i-$a-c24; wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-c24.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a c24 rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
for n in 7e 7eM; do git -C /root/lane/memra worktree remove --force /root/lane/t-$n; done; git -C /root/lane/memra worktree prune
echo "V7E_DONE $RSHA" >> $S
