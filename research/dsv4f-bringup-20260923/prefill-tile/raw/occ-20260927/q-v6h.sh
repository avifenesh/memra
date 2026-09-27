#!/usr/bin/env bash
# q-v6h.sh (SE pair): the prefill FP8 tile, round 2 around the 8x4 winner on lane $1. Box-local builds change only
# the five tile defines (TT TN SUB DEC MINB); each runs the kernel-boundary bit test
# dsv4_gemm_tile_gpu (every output bit against the GEMV loop) under nsys on one card, and the tile
# kernel's summed device time is the screen. Order B C D E G H H G E D C B.
set -u
TSHA=$1
S=/root/rcpt/q-v6h.summary; R=/root/rcpt/tile-occ-v6h; mkdir -p $R
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-prefill-tile-occ-20260927
git worktree add -f /root/lane/t-tile $TSHA > /dev/null 2>&1; git -C /root/lane/t-tile checkout -q --detach $TSHA
F=/root/lane/t-tile/crates/memra-engine/cu/dsv4_gpu.cu
[[ -d /root/lane/target-tile ]] || cp -a /root/lane/target-main13 /root/lane/target-tile
for v in E:8:4:8:1:4 I:8:4:4:1:5 J:8:2:8:1:6 K:4:4:8:1:6 L:8:4:8:0:4 M:16:2:8:1:4; do
  IFS=: read n tt tn sub dec minb <<< "$v"
  git -C /root/lane/t-tile checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
  sed -i "s/^#define DSV4_TILE_TT 8$/#define DSV4_TILE_TT $tt/; s/^#define DSV4_TILE_TN 8$/#define DSV4_TILE_TN $tn/; s/^#define DSV4_TILE_SUB 8$/#define DSV4_TILE_SUB $sub/; s/^#define DSV4_TILE_DEC 0$/#define DSV4_TILE_DEC $dec/; s/^#define DSV4_TILE_MINB 1$/#define DSV4_TILE_MINB $minb/" $F
  git -C /root/lane/t-tile diff > $R/variant-$n.diff
  ( cd /root/lane/t-tile && CARGO_TARGET_DIR=/root/lane/target-tile cargo test --release -p memra-engine --test dsv4_gemm_tile_gpu --no-run -j 56 > $R/build-$n.log 2>&1 )
  TB=$(grep -oE '/root/lane/target-tile/release/deps/dsv4_gemm_tile_gpu-[0-9a-f]+' $R/build-$n.log | head -1)
  mkdir -p /root/lane/bin-tileR$n && cp $TB /root/lane/bin-tileR$n/tiletest
  echo "build tile$n TT=$tt TN=$tn SUB=$sub DEC=$dec MINB=$minb $(tail -1 $R/build-$n.log | cut -c1-80) diff_lines $(wc -l < $R/variant-$n.diff)" >> $S
done
git -C /root/lane/t-tile checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
i=0
for a in E I J K L M M L K J I E; do
  i=$((i+1)); d=$R/run$i-$a; mkdir -p $d
  env CUDA_VISIBLE_DEVICES=0 timeout 900 nsys profile -t cuda -o $d/prof -f true /root/lane/bin-tileR$a/tiletest --ignored --test-threads=1 9>&- > $d/test.log 2>&1
  rc=$?
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/prof.nsys-rep > /dev/null 2>&1
  tile=$(python3 -c "import csv,sys; r=[x for x in csv.DictReader(open('$d/kern_cuda_gpu_kern_sum.csv')) if 'dsv4_gemm_fp8_tile_kernel' in x['Name']]; print('%.1f ms over %d launches' % (sum(float(x['Total Time (ns)']) for x in r)/1e6, sum(int(x['Instances']) for x in r)))" 2>/dev/null)
  echo "run$i $a rc=$rc $(grep -hE 'test result|panicked' $d/test.log | tail -1 | cut -c1-80) | tile $tile" >> $S
  rm -f $d/prof.nsys-rep $d/prof.sqlite
done
echo "V6H_DONE $TSHA" >> $S
