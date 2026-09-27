#!/usr/bin/env bash
# q-s2zh.sh (second SE pair): the router mirror lane's fixed fixture count ($1): component tests.
set -u
XSHA=$1
S=/root/rcpt/q-s2zh.summary; R=/root/rcpt/xmirror-comp-s2zh; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-xmirror-route-20260927
git -C /root/lane/t-zgX checkout -q --detach $XSHA
[[ "$(git -C /root/lane/t-zgX rev-parse HEAD)" == "$XSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
( cd /root/lane/t-zgX && CARGO_TARGET_DIR=/root/lane/target-hm cargo test --release -p memra-engine --lib --no-run -j 56 > $R/test-build.log 2>&1 )
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
d=$R/component; mkdir -p $d; wait_lock
( cd /root/lane/t-zgX && exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 CARGO_TARGET_DIR=/root/lane/target-hm timeout 3600 cargo test --release -p memra-engine --lib -j 56 -- --ignored --test-threads=1 --nocapture cuda_fused cuda_mrow_stream_matches cuda_deferred_partition cuda_tp_ep_local_only 9>&- > $d/gate.log 2>&1 )
echo "component rc=$? $(grep -hE '^test result|FAILED|panicked' $d/gate.log | tr '\n' ' ' | cut -c1-200)" >> $S
cd /root/lane/memra && for w in t-zgM5 t-zgX; do git worktree remove --force /root/lane/$w > /dev/null 2>&1; done; git worktree prune
echo "S2ZH_DONE $XSHA" >> $S
