#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/serve-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
BIN=/home/avifenesh/.cache/inkling2mini/serve-frozen1
M=/home/avifenesh/models/inkling2-mini
O=$L/s1
mkdir -p $O; cd $O
cat $BIN/SHA256SUMS
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; grep -v '^ink2-gpu: class\|^ink2-gpu: memory' $tag.log | tail -n 4; }
run gate $BIN/ink2-gpu-gate
run p1-float $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --out p1-float.json
run p1-declared $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out p1-declared.json
run batch-float $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-float.json
run batch-declared $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-declared.json
run batch-declared-p2 $BIN/ink2-gpu $M --class declared --tape $T/p2_window.exact_f32 --batch-check 6,24,5 --out batch-declared-p2.json
