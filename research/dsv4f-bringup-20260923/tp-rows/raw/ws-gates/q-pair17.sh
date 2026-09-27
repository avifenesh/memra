#!/usr/bin/env bash
# q-pair17.sh (2x RTX PRO 6000 WS pod): memra #710 B-row lane 504d0e1f2, the multi-row
# dense-fast FP8 GEMV. 1) dsv4_dense_fast_rows_gpu bit test. 2) dsv4_rows_gate on TP/EP and on
# PP-2 (identity + timing). 3) The DSpark gate on TP/EP and PP-2 (verify rounds ride the
# kernel), proposal shas against the #720 receipts. 4) TP/EP B-row anatomy at B=2 and 4.
set -u
S=/root/rcpt/q-pair17.summary; R=/root/rcpt/tp-rows-fast; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json
cd /root/lane/memra && git fetch -q origin lane/dsv4-tp-rows-20260925
git -C /root/lane/t-rows checkout -q --detach 504d0e1f2
( cd /root/lane/t-rows && CARGO_TARGET_DIR=/root/lane/target-rows cargo build --release -j 56 -p memra-engine --bins && \
  CARGO_TARGET_DIR=/root/lane/target-rows cargo test --release -j 56 -p memra-engine --test dsv4_dense_fast_rows_gpu --no-run ) > $R/build.log 2>&1
echo "build rows $(git -C /root/lane/t-rows rev-parse --short HEAD) rc=$?" >> $S
X=/root/lane/target-rows/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 "$@" 9>&- > $d/gate.log 2>&1 ); }
locked $R/unit bash -c "cd /root/lane/t-rows && CARGO_TARGET_DIR=/root/lane/target-rows cargo test --release -p memra-engine --test dsv4_dense_fast_rows_gpu -- --ignored --test-threads=1"
echo "unit rc=$? $(grep -hE 'cases bit-identical|red arm|test result|panicked' $R/unit/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $X/dsv4_rows_gate $M /root/box/tape-rebuild.txt 24 64
echo "rows-tpep rc=$? $(grep -hE 'PASS|FAILED|DIVERGENCE|TIME rep=1|panicked' $R/rows-tpep/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
locked $R/rows-pp $X/dsv4_rows_gate $M /root/box/tape-rebuild.txt 24 64
echo "rows-pp rc=$? $(grep -hE 'PASS|FAILED|DIVERGENCE|TIME rep=1|panicked' $R/rows-pp/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
for topo in tpep pp; do
  d=$R/dspark-$topo; mkdir -p $d; wait_lock
  if [[ $topo == pp ]]; then
    ( export MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device; /root/box/gate.sh $d $X/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served )
  else
    ( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $X/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
  fi
  echo "dspark-$topo $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|FAIL|panicked|GATE_DONE' $d/gate.log | head -n 4 | cut -c1-160 | tr '\n' ' ')" >> $S
done
exec 9>/tmp/memra-gpu.lock
while ! flock -n 9; do sleep 10; done
for b in 2 4; do
  d=$R/anat-b$b; mkdir -p $d
  ( NVIDIA_TF32_OVERRIDE=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_PROFILE=$b nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt -o $d/nsys -f true $X/dsv4_rows_gate $M /root/box/tape-rebuild.txt 24 64 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 64 > $d/ana.txt 2>&1
  rm -f $d/nsys.sqlite $d/nsys.nsys-rep
  echo "anat B=$b $(grep -hE 'PROFILE' $d/gate.log | cut -c1-100) | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
echo PAIR17_DONE >> $S
