#!/usr/bin/env bash
# q-v6p.sh (SE pair): the tile lane after its knobs became constants ($1): the tile bit test, the
# tile kernel sums against the gated build (T, bin-tiletest-T) in order T T2 T2 T, and the long
# gate hash. Builds in target-main13 once q-v6m is done with it.
set -u
TSHA=$1
S=/root/rcpt/q-v6p.summary; R=/root/rcpt/tile-const-v6p; mkdir -p $R
until grep -q V6M_DONE /root/rcpt/q-v6m.summary 2>/dev/null; do sleep 60; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
cd /root/lane/memra && git fetch -q origin lane/dsv4-prefill-tile-occ-20260927
git -C /root/lane/t-tile checkout -q --detach $TSHA
[[ "$(git -C /root/lane/t-tile rev-parse HEAD)" == "$TSHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
( cd /root/lane/t-tile && CARGO_TARGET_DIR=/root/lane/target-main13 cargo build --release -j 56 -p memra-engine --bin dsv4_tp_replay_long_gate > $R/build.log 2>&1 && CARGO_TARGET_DIR=/root/lane/target-main13 cargo test --release -p memra-engine --test dsv4_gemm_tile_gpu --no-run -j 56 >> $R/build.log 2>&1 )
echo "build rc=$?" >> $S
TB=$(grep -oE '/root/lane/target-main13/release/deps/dsv4_gemm_tile_gpu-[0-9a-f]+' $R/build.log | tail -1)
mkdir -p /root/lane/bin-tiletest-T2; cp $TB /root/lane/bin-tiletest-T2/tile_test; cp /root/lane/target-main13/release/dsv4_tp_replay_long_gate /root/lane/bin-tiletest-T2/
sha256sum /root/lane/bin-tiletest-T/tile_test /root/lane/bin-tiletest-T2/* > $R/binaries.sha256
/usr/local/cuda/bin/cuobjdump -res-usage /root/lane/bin-tiletest-T2/tile_test 2>/dev/null | grep -A1 'dsv4_gemm_fp8_tile\|dsv4_dots_f32acc_tile' | grep -oE 'Function [^:]*|REG:[0-9]+' | paste - - >> $S
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
d=$R/bits; mkdir -p $d
env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 timeout 900 /root/lane/bin-tiletest-T2/tile_test --ignored --nocapture --test-threads=1 9>&- > $d/gate.log 2>&1
echo "bits T2 rc=$? $(grep -hE '^test result|FAILED|panicked|EXACT' $d/gate.log | tr '\n' ' ' | cut -c1-300)" >> $S
i=0
for a in T T2 T2 T; do
  i=$((i+1)); d=$R/run$i-$a; mkdir -p $d
  env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 timeout 900 nsys profile -t cuda -o $d/prof -f true /root/lane/bin-tiletest-$a/tile_test --ignored --test-threads=1 9>&- > $d/test.log 2>&1
  rc=$?
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/prof.nsys-rep > /dev/null 2>&1
  sums=$(python3 -c "
import csv
r=list(csv.DictReader(open('$d/kern_cuda_gpu_kern_sum.csv')))
for name in ('dsv4_gemm_fp8_tile_kernel','dsv4_dots_f32acc_tile_kernel'):
    x=[k for k in r if name in k['Name']]
    print('%s %.1f ms over %d' % (name, sum(float(k['Total Time (ns)']) for k in x)/1e6, sum(int(k['Instances']) for k in x)), end=' | ')
" 2>/dev/null)
  echo "run$i $a rc=$rc $(grep -hE 'test result|panicked' $d/test.log | tail -1 | cut -c1-40) | $sums" >> $S
  rm -f $d/prof.nsys-rep $d/prof.sqlite
done
d=$R/long-304; mkdir -p $d
env NVIDIA_TF32_OVERRIDE=0 timeout 3600 /root/lane/bin-tiletest-T2/dsv4_tp_replay_long_gate $M $T 304 9>&- > $d/gate.log 2>&1
echo "long-304 T2 rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | sort -u | tr '\n' ' ')" >> $S
echo "V6P_DONE $TSHA" >> $S
