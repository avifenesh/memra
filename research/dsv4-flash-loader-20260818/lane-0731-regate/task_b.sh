#!/bin/bash
# 0731 re-gate task B — the f32x extension rung (owner-authorized, conditional on the
# quality bar). Gate order per the banked derivation; every gate logs independently.
cd /home/ubuntu/memra-src
B=target/release
OUT=/home/ubuntu/dsv4-0731-regate-out
MODEL=/home/ubuntu/models/dsv4-flash-0731-nvfp4
FIX=/home/ubuntu/dsv4-oracle0731/fixtures-mint
TF=/home/ubuntu/dsv4-oracle0731/greedy/cpu_greedy_ref.tf.json
GREEDYJ=/home/ubuntu/dsv4-oracle0731/greedy/cpu_greedy_ref.json
export MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DECODE_PATH=device
set -x
# (iii) f32x decode-gate 260 (a1/c/d/e)
MEMRA_DSV4_DOTS_ARM=f32x $B/dsv4-gpu-decode-gate $MODEL $FIX/dsv4_0731_fixtures_ref.json $GREEDYJ $OUT/dec-f32x 260 0,1 > $OUT/decode-gate-0731-f32x.log 2>&1
echo "decode-gate-0731-f32x exit $?" >> $OUT/status.txt
# (v) f32x tf-gate 160 vs the banked trajectory (runs before the slow CPU verify;
# order within task B does not gate order of evidence)
MEMRA_DSV4_DOTS_ARM=f32x $B/dsv4-gpu-tf-gate $MODEL $TF $OUT/tf-f32x 0,1 > $OUT/tf-gate-0731-f32x.log 2>&1
echo "tf-gate-0731-f32x exit $?" >> $OUT/status.txt
# informational single-run ms/step delta: f32 vs f32x on the mint (NOT an A/B claim)
MEMRA_DSV4_DOTS_ARM=f32 $B/dsv4-decode-bench $MODEL $FIX/dsv4_0731_fixtures_ref.json $OUT/bench-0731-f32-1024.json 1024 0,1 > $OUT/bench-0731-f32-1024.log 2>&1
echo "bench-0731-f32 exit $?" >> $OUT/status.txt
MEMRA_DSV4_DOTS_ARM=f32x $B/dsv4-decode-bench $MODEL $FIX/dsv4_0731_fixtures_ref.json $OUT/bench-0731-f32x-1024.json 1024 0,1 > $OUT/bench-0731-f32x-1024.log 2>&1
echo "bench-0731-f32x exit $?" >> $OUT/status.txt
# (iv) CPU teacher-forcing (260-class) over the f32x decode trajectory
$B/dsv4-greedy-verify $MODEL $OUT/dec-f32x/decode_seq_for_verify.json $OUT/dec-f32x/cpu-verify > $OUT/cpu-verify-dec-f32x.log 2>&1
echo "cpu-verify-dec-f32x exit $?" >> $OUT/status.txt
echo TASK_B_DONE >> $OUT/status.txt
