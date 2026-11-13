# r2 on wrx: build, then a two-card confirm on cards 2+3 (after CUTLASS's)

r2 = the router branch's r2 commit (the fp32 FMA family one chain an output, the narrow output's chain through the
ring kernel, the compressor's pair in one launch). Reference arm already on wrx: `perf-bins/kpc` (the current route).
About 12 minutes of build (no GPU, no timed cell meanwhile), then about 6 + 9 minutes on cards 2+3.

1. Laptop: the tree, the build helper and the byte script.

       rsync -a --delete --exclude target --exclude .git --link-dest=/home/access/dsv41/src-gpu \
         ~/.cache/dsv41-memra/wt-router/ wrx:/home/access/dsv41/src-r2/
       scp ~/.cache/dsv41-memra/r1-dig/wrx/build_one.sh ~/.cache/dsv41-memra/r1-dig/wrx/r2_bytes.sh \
         wrx:/home/access/dsv41/bin/

2. wrx build, no GPU (bin/wrx_build.sh: cores 26-31,58-63, 4 jobs, nice 19; ~12 min):

       ssh wrx '(cd /home/access/dsv41 && nohup bash bin/wrx_build.sh r2-build bash bin/build_one.sh r2) \
         > /home/access/dsv41/receipts-perf-build-r2.out 2>&1 < /dev/null & echo started'
       waitfor --ssh wrx 'grep -q "BUILD-DONE r2" /home/access/dsv41/receipts-perf-build-r2.out' --timeout 30m

3. Byte gate vs kpc, one card of the pair (~6 min): the layer gate's 111 saved tensors of kpc and r2 compared
   file by file (want 0 differ), then dsv41-bench gemm kpc r2 r2 kpc (router output hashes at 1/4/5/8 rows equal):

       ssh wrx 'cd /home/access/dsv41 && bash bin/perf_runc.sh r2-bytes/lease 600 2 bash bin/r2_bytes.sh; \
         cat receipts-perf/r2-bytes/bytes.txt'

4. Two-card step timing (~9 min): r2's tp gates on the pair (default, peer-pull), then decode 1 row and the 4- and
   8-row blocks, kpc against r2, A B B A B A A B, 20 layers, steady figures (from 0.5 s busy):

       ssh wrx 'cd /home/access/dsv41 && K3_ARM_A=kpc K3_ARM_B=r2 K3_TAG=r2-pair23 K3_CARDS=23 K3_DEV=0,1 \
         K3_LAYERS=20 K3_ORDER="A B B A B A A B" bash bin/perf_k3_four.sh 600 > receipts-perf-r2-pair23.out 2>&1; \
         tail -n 20 receipts-perf-r2-pair23.out'

   For the comparison with r1 on the same cards (optional, ~5 min): `K3_ARM_A=r1 K3_ARM_B=r2 K3_TAG=r2-r1-pair23
   K3_GATES=0 K3_ORDER="A B B A"` with the rest as above. The reference: r1 vs kpc on this pair was measured only
   with a build overlapping (contaminated, `r1-pair23-buildoverlap`): 6.665 / 6.354 ms one row.

Four cards later (same window form as block-1113 item 4): `K3_ARM_A=kpc K3_ARM_B=r2 K3_TAG=r2-four K3_CARDS=0123
K3_DEV=0,1,2,3 K3_LAYERS=40 K3_ORDER="A B B A B A A B" bash bin/perf_k3_four.sh 900`.

What to expect, from the RTX 5090 and GB10 kernels (not yet measured on the PRO 6000): the ring's router call is
1.5 us (kernel) to 4.4 us (engine timer) above r1's at 1 to 4 rows and 13 us below it at 8 rows, against 24 to 43 us
below kpc's, and the compressor's pair is one launch. Outputs equal to kpc's: 111 of 111 tensors, the router hashes,
the tpbench decode gen hashes; tp gates 480/0 and 478/0 on the pair.
