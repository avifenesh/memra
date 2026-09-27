#!/usr/bin/env bash
# q-v6o.sh (SE pair): the tile bit test under nsys on one card, main's binary (M) against the
# rebased tile lane's (T), order M T T M: the FP8 tile and the dots tile kernel sums of each.
set -u
S=/root/rcpt/q-v6o.summary; R=/root/rcpt/tile-vs-main-v6o; mkdir -p $R
until grep -q V6L_DONE /root/rcpt/q-v6l.summary 2>/dev/null; do sleep 60; done
export PATH=/usr/local/cuda/bin:$PATH
sha256sum /root/lane/bin-tiletest-*/tile_test > $R/binaries.sha256
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
i=0
for a in M T T M; do
  i=$((i+1)); d=$R/run$i-$a; mkdir -p $d
  env NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=0 timeout 900 nsys profile -t cuda -o $d/prof -f true /root/lane/bin-tiletest-$a/tile_test --ignored --test-threads=1 9>&- > $d/test.log 2>&1
  rc=$?
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/prof.nsys-rep > /dev/null 2>&1
  sums=$(python3 -c "
import csv
r=list(csv.DictReader(open('$d/kern_cuda_gpu_kern_sum.csv')))
for name in ('dsv4_gemm_fp8_tile_kernel','dsv4_dots_f32acc_tile_kernel'):
    x=[k for k in r if name in k['Name']]
    print('%s %.1f ms over %d' % (name.split('_')[1] + '_' + name.split('_')[2], sum(float(k['Total Time (ns)']) for k in x)/1e6, sum(int(k['Instances']) for k in x)), end=' | ')
" 2>/dev/null)
  echo "run$i $a rc=$rc $(grep -hE 'test result|panicked' $d/test.log | tail -1 | cut -c1-60) | $sums" >> $S
  rm -f $d/prof.nsys-rep $d/prof.sqlite
done
echo "V6O_DONE" >> $S
