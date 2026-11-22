#!/bin/bash
# batched == solo of both classes on one ink2-gpu (BIN), receipts to OUT.
L=/home/avifenesh/.cache/inkling2mini
T=$L/oracle/tapes
M=/home/avifenesh/models/inkling2-mini
mkdir -p $OUT; cd $OUT
cat $BIN/COMMIT $BIN/SHA256SUMS
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; tail -n 2 $tag.log; }
for cls in shipped nvfp4-rig; do
  run batch-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-$cls.json
  run batch-long-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --batch-check 4,16,256,1000,60 --out batch-long-$cls.json
done
echo BATCH_ONLY_DONE
