#!/usr/bin/env bash
# q-v6g.sh (SE pair): the prefill FP8 tile's register diet on lane $1. Box-local builds change only
# the five tile defines (TT TN SUB DEC MINB); each runs the kernel-boundary bit test
# dsv4_gemm_tile_gpu (every output bit against the GEMV loop) under nsys on one card, and the tile
# kernel's summed device time is the screen. Order B C D E G H H G E D C B.
set -u
TSHA=$1
S=/root/rcpt/q-v6g.summary; R=/root/rcpt/tile-occ-v6g; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-prefill-tile-occ-20260927
git worktree add -f /root/lane/t-tile $TSHA > /dev/null 2>&1; git -C /root/lane/t-tile checkout -q --detach $TSHA
F=/root/lane/t-tile/crates/memra-engine/cu/dsv4_gpu.cu
[[ -d /root/lane/target-tile ]] || cp -a /root/lane/target-main13 /root/lane/target-tile
for v in B:8:8:8:0:1 C:8:8:4:1:3 D:8:8:2:1:4 E:8:4:8:1:4 G:16:4:4:1:3 H:8:8:4:0:3; do
  IFS=: read n tt tn sub dec minb <<< "$v"
  git -C /root/lane/t-tile checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
  sed -i "s/^#define DSV4_TILE_TT 8$/#define DSV4_TILE_TT $tt/; s/^#define DSV4_TILE_TN 8$/#define DSV4_TILE_TN $tn/; s/^#define DSV4_TILE_SUB 8$/#define DSV4_TILE_SUB $sub/; s/^#define DSV4_TILE_DEC 0$/#define DSV4_TILE_DEC $dec/; s/^#define DSV4_TILE_MINB 1$/#define DSV4_TILE_MINB $minb/" $F
  git -C /root/lane/t-tile diff > $R/variant-$n.diff
  ( cd /root/lane/t-tile && CARGO_TARGET_DIR=/root/lane/target-tile cargo test --release -p memra-engine --test dsv4_gemm_tile_gpu --no-run -j 56 > $R/build-$n.log 2>&1 )
  TB=$(grep -oE '/root/lane/target-tile/release/deps/dsv4_gemm_tile_gpu-[0-9a-f]+' $R/build-$n.log | head -1)
  mkdir -p /root/lane/bin-tile$n && cp $TB /root/lane/bin-tile$n/tiletest
  echo "build tile$n TT=$tt TN=$tn SUB=$sub DEC=$dec MINB=$minb $(tail -1 $R/build-$n.log | cut -c1-80) diff_lines $(wc -l < $R/variant-$n.diff)" >> $S
done
git -C /root/lane/t-tile checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
i=0
for a in B C D E G H H G E D C B; do
  i=$((i+1)); d=$R/run$i-$a; mkdir -p $d
  env CUDA_VISIBLE_DEVICES=0 timeout 900 nsys profile -t cuda -o $d/prof -f true /root/lane/bin-tile$a/tiletest --ignored --test-threads=1 9>&- > $d/test.log 2>&1
  rc=$?
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/prof.nsys-rep > /dev/null 2>&1
  tile=$(grep -h 'dsv4_gemm_fp8_tile_kernel' $d/kern_cuda_gpu_kern_sum.csv 2>/dev/null | awk -F'","' '{gsub(/"/,"",$2); s+=$2; c+=$3} END {printf "%.3f ms over %d launches", s/1e6, c}')
  echo "run$i $a rc=$rc $(grep -hE 'test result|panicked' $d/test.log | tail -1 | cut -c1-80) | tile $tile" >> $S
  rm -f $d/prof.nsys-rep $d/prof.sqlite
done
echo "V6G_DONE $TSHA" >> $S
