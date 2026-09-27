#!/usr/bin/env bash
# q-v6l.sh (SE pair): q-v6k ran the tile bit test without --ignored (4 ignored); this runs it.
set -u
S=/root/rcpt/q-v6l.summary; R=/root/rcpt/tile-rebased-v6k
until grep -q V6K_DONE /root/rcpt/q-v6k.summary 2>/dev/null; do sleep 60; done
TB=$(grep -oE '/root/lane/target-tile/release/deps/dsv4_gemm_tile_gpu-[0-9a-f]+' $R/test-build.log | head -1)
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
d=$R/tile-test-ignored; mkdir -p $d; wait_lock
( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 timeout 3600 $TB --ignored --nocapture 9>&- > $d/gate.log 2>&1 )
echo "tile-test --ignored rc=$? $(sha256sum $TB | cut -c1-16) $(grep -hE '^test result|FAILED|panicked|EXACT' $d/gate.log | tr '\n' ' ' | cut -c1-400)" >> $S
echo "V6L_DONE" >> $S
