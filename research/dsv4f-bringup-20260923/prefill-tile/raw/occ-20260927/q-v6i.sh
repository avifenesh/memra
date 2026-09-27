#!/usr/bin/env bash
# q-v6i.sh (SE pair): the prefill FP8 tile on the served chunked prefill, TTFT. Lane $1 builds
# with the tile defines changed box-locally: B = committed (8,8,8,LUT,1), E = (8,4,8,cvt,4),
# J = (8,2,8,cvt,6), the kernel sweep's two winners. One boot per row B E J J E B B E J (N=3),
# cells-ttft (c1 greedy, c2 at 1500 words, c1 at 12000 words).
set -u
TSHA=$1
S=/root/rcpt/q-v6i.summary; R=/root/rcpt/tile-ttft-v6i; mkdir -p $R
until grep -q V6H_DONE /root/rcpt/q-v6h.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 16 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
F=/root/lane/t-tile/crates/memra-engine/cu/dsv4_gpu.cu
git -C /root/lane/t-tile checkout -q --detach $TSHA
for v in B:8:8:8:0:1 E:8:4:8:1:4 J:8:2:8:1:6; do
  IFS=: read n tt tn sub dec minb <<< "$v"
  git -C /root/lane/t-tile checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
  sed -i "s/^#define DSV4_TILE_TT 8$/#define DSV4_TILE_TT $tt/; s/^#define DSV4_TILE_TN 8$/#define DSV4_TILE_TN $tn/; s/^#define DSV4_TILE_SUB 8$/#define DSV4_TILE_SUB $sub/; s/^#define DSV4_TILE_DEC 0$/#define DSV4_TILE_DEC $dec/; s/^#define DSV4_TILE_MINB 1$/#define DSV4_TILE_MINB $minb/" $F
  git -C /root/lane/t-tile diff > $R/variant-$n.diff
  bash /root/box/build.sh /root/lane/t-tile /root/lane/target-tile tilesrv$n
  mkdir -p /root/lane/bin-tsrv$n && cp /root/lane/target-tile/release/memra-server /root/lane/bin-tsrv$n/
  echo "build tilesrv$n $(grep -hE 'EXIT' /root/build-tilesrv$n.log | tr '\n' ' ') diff_lines $(wc -l < $R/variant-$n.diff) $(sha256sum /root/lane/bin-tsrv$n/memra-server | cut -c1-16)" >> $S
done
git -C /root/lane/t-tile checkout -q -- crates/memra-engine/cu/dsv4_gpu.cu
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
i=0
for a in B E J J E B B E J; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d /root/lane/bin-tsrv$a/memra-server /root/box/cells-ttft.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "V6I_DONE $TSHA" >> $S
