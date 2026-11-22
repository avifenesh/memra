#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; K=/home/avifenesh/.cache/inkling2mini/klsens/corpus
O=$P/set5; mkdir -p $O
B=$P/a23-bin/ink2-gpu
A='mix=f32;mix=f32,epilogue=f32;epilogue=f32'
for c in rawdoc chatwrap; do
  $B $M --class float --tape $T/p1_chat.exact_f32 --context 2048 --corpus $K/$c.json --arms "$A" --out $O/$c.json > $O/$c.log 2>&1; echo RC_$c=$?
done
for c in p1 p2; do
  $B $M --class float --tape $T/p1_chat.exact_f32 --context 2048 --corpus $P/corpus-$c.json --arms "$A" --out $O/$c.json > $O/$c.log 2>&1; echo RC_$c=$?
done
for c in L1 L2; do
  $B $M --class float --tape $T/p1_chat.exact_f32 --context 8704 --corpus $P/corpus-$c.json --arms "$A" --out $O/$c.json > $O/$c.log 2>&1; echo RC_$c=$?
done
bash $P/cls2.sh gm2-rig nvfp4-rig $P/a23-bin "" $P/a23-bin "--prec mix=f32,epilogue=f32" > $P/gm2-rig.log 2>&1; echo RC_t=$?
