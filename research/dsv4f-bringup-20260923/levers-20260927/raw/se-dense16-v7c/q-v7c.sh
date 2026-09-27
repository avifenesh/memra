#!/usr/bin/env bash
# q-v7c.sh (SE pair): dense-fast M-row launches up to 16 rows ($2) against the B-row lane it
# stacks on ($1). Component tests (dense-fast rows at 2..16, grouped wo_a), long gate hash,
# TP/EP rows, wide 16, the rows gate's wide-16 timing B D D B, served cells-c24 B D D B.
set -u
BSHA=$1; DSHA=$2
S=/root/rcpt/q-v7c.summary; R=/root/rcpt/dense16-v7c; mkdir -p $R
until grep -q V7B_DONE /root/rcpt/q-v7b.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 40 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
X=/root/lane/target-main13/release
cd /root/lane/memra && git fetch -q origin lane/dsv4-brow-diet-20260927 lane/dsv4-dense16-20260927
for v in B:$BSHA D:$DSHA; do IFS=: read n sha <<< "$v"
  git worktree add -f /root/lane/t-7c$n $sha > /dev/null 2>&1; git -C /root/lane/t-7c$n checkout -q --detach $sha
  [[ "$(git -C /root/lane/t-7c$n rev-parse HEAD)" == "$sha" ]] || { echo "TREE_MISMATCH $n" >> $S; exit 1; }
  bash /root/box/build.sh /root/lane/t-7c$n /root/lane/target-main13 7c$n
  mkdir -p /root/lane/bin-7c$n; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate /root/lane/bin-7c$n/
  echo "build $n $(git -C /root/lane/t-7c$n rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-7c$n.log | tr '\n' ' ')" >> $S
done
( cd /root/lane/t-7cD && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --lib --test dsv4_dense_fast_rows_gpu --no-run -j 56 > $R/test-build.log 2>&1 )
sha256sum /root/lane/bin-7c*/* > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
( cd /root/lane/t-7cD && locked $R/component env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --test dsv4_dense_fast_rows_gpu -j 56 -- --ignored --test-threads=1 --nocapture )
echo "component rows rc=$? $(grep -hE '^test result|FAILED|panicked|^PASS' $R/component/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
( cd /root/lane/t-7cD && locked $R/component-grouped env CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_gemv_fp8_grouped )
echo "component grouped rc=$? $(grep -hE '^test result|FAILED|panicked' $R/component-grouped/gate.log | tr '\n' ' ' | cut -c1-200)" >> $S
XD=/root/lane/bin-7cD
locked $R/long-304 $XD/dsv4_tp_replay_long_gate $M $T 304
echo "long-304 D rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XD/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
bin() { case $1 in B) echo /root/lane/bin-7cB ;; *) echo /root/lane/bin-7cD ;; esac; }
i=0
for a in D B D B; do
  i=$((i+1)); d=$R/wide16-t$i-$a
  locked $d env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=16 $(bin $a)/dsv4_rows_gate $M $T 24 64
  echo "wide16 t$i $a rc=$? $(grep -hE '^PASS|FAIL|panicked' $d/gate.log | cut -c1-70 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in B D D B; do
  i=$((i+1)); d=$R/r$i-$a-c24; wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-c24.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a c24 rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
for n in B D; do git -C /root/lane/memra worktree remove --force /root/lane/t-7c$n; done; git -C /root/lane/memra worktree prune
echo "V7C_DONE $DSHA" >> $S
