#!/usr/bin/env bash
# q-v3l.sh (2x RTX PRO 6000 Server Edition), after q-v3k.sh: served TTFT A/B for the prefill dense
# tile and compressor dots tile (memra #472, #700), lane/dsv4-prefill-gemm-20260924 7e15b241d against main a8d0121d9, one boot per
# row, order L B B L L B (N=3), cells-ttft.txt (8k and 32k prompts, 64 tokens, text kept).
while ! grep -q V3K_DONE /root/rcpt/q-v3k.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3l.summary; R=/root/rcpt/ttft; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-prefill-gemm-20260924 main
for a in pgemm:7e15b241d main9:a8d0121d9; do
  name=${a%%:*}; c=${a##*:}
  git worktree add -f /root/lane/t-$name $c > /dev/null 2>&1; git -C /root/lane/t-$name checkout -q --detach $c
  [[ -d /root/lane/target-$name ]] || cp -a /root/lane/target-rows /root/lane/target-$name
  bash /root/box/build.sh /root/lane/t-$name /root/lane/target-$name $name
  echo "build $name $(git -C /root/lane/t-$name rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-$name.log | tr '\n' ' ') server $(sha256sum /root/lane/target-$name/release/memra-server | cut -c1-16)" >> $S
done
i=0
for arm in L B B L L B; do
  i=$((i+1)); d=$R/r$i-$arm; bin=/root/lane/target-pgemm/release/memra-server; [[ $arm == B ]] && bin=/root/lane/target-main9/release/memra-server
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  /root/box/cell.sh $d $bin /root/box/cells-ttft.txt MEMRA_TIMEOUT_MS_MAX=900000 MEMRA_DSV4_SESSIONS=1 > $d.out 2>&1
  echo "ttft r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo V3L_DONE >> $S
