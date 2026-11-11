#!/usr/bin/env bash
# GB10, one lease (<= 30 min; step 4 the nsys of one layer-2 rep): bs1f (the small-row ring sized by the call, 6 stages; same chain program) against bs1:
#   1. kernel gate (with the partial-N cases)
#   2. layer gate --save: the 111 saved tensors must equal bs1's set byte for byte
#   3. dsv41-bench gemm, A B B A (A = bs1, B = bs1f): the decode and prefill shapes
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/verify-bs1f; mkdir -p $O; : > $O/runs.txt
export MEMRA_FULL_PREC=1
B=$R/bins/bs1f/dsv41-gate
timeout 5m $B kernels $D/fixtures-gpu/rand --out $O/kernels.json > $O/kernels.log 2>&1
echo "kernels rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/kernels.log | tail -1) $(date +%T)" >> $O/runs.txt
grep -a "^FAIL" $O/kernels.log | head -10 >> $O/runs.txt
rm -rf $R/save-bs1f; mkdir -p $R/save-bs1f
timeout 9m $B layer $C $D/captures-v2/needle --layers 1,2,14,20,21,24 --decode 2 --token-map $TM \
  --floor $D/captures-v2/needle-floor --save $R/save-bs1f --out $O/layer.json > $O/layer.log 2>&1
echo "layer rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/layer.log | tail -1) $(date +%T)" >> $O/runs.txt
n=0; bad=0
for f in $(cd $R/save-bs1 && find . -name '*.bin' | sort); do n=$((n+1)); cmp -s $R/save-bs1/$f $R/save-bs1f/$f || bad=$((bad+1)); done
echo "saved tensors against bs1's set: $n compared, $bad differ" >> $O/runs.txt
i=0
for a in A B B A; do
  i=$((i+1)); BD=$R/bins/bs1; [ $a = B ] && BD=$R/bins/bs1f
  timeout 3m $BD/dsv41-bench gemm --reps 20 --out $O/run$i-$a-gemm.json > $O/run$i-$a-gemm.log 2>&1
  echo "run$i-$a gemm rc $? $(date +%T)" >> $O/runs.txt
done
# 4. one layer-2 rep at 4,096 tokens under nsys (bs1f, --nvtx: a range per op), then its kernel and NVTX sums
timeout 6m /usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none -f true -o $O/layer2-4k \
  $R/bins/bs1f/dsv41-bench layer $HOME/models/DeepSeek-V4.1-Flash-1017 --ids $R/prefill-4096.i32 --token-map $TM \
  --layers 2 --prefix 4096 --reps 1 --nvtx --out $O/layer2-4k.json > $O/layer2-4k.log 2>&1
echo "nsys rc $? $(date +%T)" >> $O/runs.txt
timeout 4m /usr/local/cuda/bin/nsys stats -r cuda_gpu_kern_sum,nvtx_sum -f csv -o $O/layer2-4k $O/layer2-4k.nsys-rep > $O/layer2-4k-stats.log 2>&1
echo "nsys stats rc $? $(date +%T)" >> $O/runs.txt
echo "VERIFY-BS1F-DONE $(date +%T)" >> $O/runs.txt
