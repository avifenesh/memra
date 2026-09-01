#!/bin/bash
# 0731 re-gate task A (publish item 2). Every gate logs independently; the script
# never aborts early — a dead orchestrator must not kill compute (spot + API weather).
cd /home/ubuntu/memra-src
B=target/release
OUT=/home/ubuntu/dsv4-0731-regate-out
MODEL=/home/ubuntu/models/dsv4-flash-0731-nvfp4
PREVIEW=/home/ubuntu/models/dsv4-flash-nvfp4
FIX=/home/ubuntu/dsv4-oracle0731/fixtures-mint
TF=/home/ubuntu/dsv4-oracle0731/greedy/cpu_greedy_ref.tf.json
GREEDYJ=/home/ubuntu/dsv4-oracle0731/greedy/cpu_greedy_ref.json
export MEMRA_DSV4_EXPERT_ARM=native
set -x
# A1 output-sample REF on the mint (native arm)
$B/dsv4-gpu-gate $MODEL $FIX/dsv4_0731_fixtures_ref.json 0,1 > $OUT/gpu-gate-0731-ref-native.log 2>&1
echo "gpu-gate-0731-ref-native exit $?" >> $OUT/status.txt
# A1b clamp cross-receipt on the mint
$B/dsv4-gpu-gate $MODEL $FIX/dsv4_0731_fixtures_artifactvariant.json 0,1 > $OUT/gpu-gate-0731-clamp-native.log 2>&1
echo "gpu-gate-0731-clamp-native exit $?" >> $OUT/status.txt
# A1c preview witness (shared load-path touched): values must match the lane-9 banked table
$B/dsv4-gpu-gate $PREVIEW /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json 0,1 > $OUT/gpu-gate-preview-witness.log 2>&1
echo "gpu-gate-preview-witness exit $?" >> $OUT/status.txt
# A2 tf-gate: shipping stack (device, dots f64 default), then the f32 dots arm
export MEMRA_DSV4_DECODE_PATH=device
$B/dsv4-gpu-tf-gate $MODEL $TF $OUT/tf-f64 0,1 > $OUT/tf-gate-0731-f64.log 2>&1
echo "tf-gate-0731-f64 exit $?" >> $OUT/status.txt
MEMRA_DSV4_DOTS_ARM=f32 $B/dsv4-gpu-tf-gate $MODEL $TF $OUT/tf-f32 0,1 > $OUT/tf-gate-0731-f32.log 2>&1
echo "tf-gate-0731-f32 exit $?" >> $OUT/status.txt
# A3 decode gate 260 (boundary short-probe), both arms
$B/dsv4-gpu-decode-gate $MODEL $FIX/dsv4_0731_fixtures_ref.json $GREEDYJ $OUT/dec-f64 260 0,1 > $OUT/decode-gate-0731-f64.log 2>&1
echo "decode-gate-0731-f64 exit $?" >> $OUT/status.txt
MEMRA_DSV4_DOTS_ARM=f32 $B/dsv4-gpu-decode-gate $MODEL $FIX/dsv4_0731_fixtures_ref.json $GREEDYJ $OUT/dec-f32 260 0,1 > $OUT/decode-gate-0731-f32.log 2>&1
echo "decode-gate-0731-f32 exit $?" >> $OUT/status.txt
# CPU teacher-forcing (260-class) over both decode trajectories
$B/dsv4-greedy-verify $MODEL $OUT/dec-f64/decode_seq_for_verify.json $OUT/dec-f64/cpu-verify > $OUT/cpu-verify-dec-f64.log 2>&1
echo "cpu-verify-dec-f64 exit $?" >> $OUT/status.txt
$B/dsv4-greedy-verify $MODEL $OUT/dec-f32/decode_seq_for_verify.json $OUT/dec-f32/cpu-verify > $OUT/cpu-verify-dec-f32.log 2>&1
echo "cpu-verify-dec-f32 exit $?" >> $OUT/status.txt
echo TASK_A_DONE >> $OUT/status.txt
# A0-regression (appended; runs after the above): preview decode-bench 1024, both
# existing arms — stream shas must equal the lane-9 banked values exactly
# (384ecde1bcff28a3... f64 default / 1acee23f157087a4... f32): proves the f32x
# dispatch refactor + new kernels byte-inert when off, on the DECODE path.
$B/dsv4-decode-bench $PREVIEW /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json $OUT/bench-preview-f64-1024.json 1024 0,1 > $OUT/bench-preview-f64-1024.log 2>&1
echo "bench-preview-f64 exit $?" >> $OUT/status.txt
MEMRA_DSV4_DOTS_ARM=f32 $B/dsv4-decode-bench $PREVIEW /home/ubuntu/dsv4-lane2-fixtures/dsv4_fixtures_artifactvariant.json $OUT/bench-preview-f32-1024.json 1024 0,1 > $OUT/bench-preview-f32-1024.log 2>&1
echo "bench-preview-f32 exit $?" >> $OUT/status.txt
grep -h logits_sha256 $OUT/bench-preview-f64-1024.json $OUT/bench-preview-f32-1024.json >> $OUT/status.txt
echo TASK_A_ALL_DONE >> $OUT/status.txt
