#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/gpu-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
T64=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var/p3.exact_f64.chunk1024.as_f32
BIN=/home/avifenesh/.cache/inkling2mini/gpu-frozen8
M=/home/avifenesh/models/inkling2-mini
O=$L/b8
mkdir -p $O
cd $O
cat $BIN/SHA256SUMS
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; grep -v '^ink2-gpu: class\|^ink2-gpu: memory' $tag.log | tail -n 5; }
run gate $BIN/ink2-gpu-gate
run actq-gate $BIN/ink2-actq-gate
run p1-declared-forced $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --force-router --force-moe-in --out p1-declared-forced-moe.json
run p2-declared-forced $BIN/ink2-gpu $M --class declared --tape $T/p2_window.actq_f32 --score --force-router --force-moe-in --out p2-declared-forced-moe.json
run p1-declared $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out p1-declared.json
run p1-declared-vs-float $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --score --out p1-declared-vs-float.json
run p2-declared-vs-float $BIN/ink2-gpu $M --class declared --tape $T/p2_window.exact_f32 --score --chunkings 512,300 --limit 1500 --out p2-declared-vs-float.json
run p3-declared-vs-f64 $BIN/ink2-gpu $M --class declared --tape $T64 --context 8448 --score --out p3-declared-vs-f64.json
run atomic-declared $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --inject-failure 33 --limit 57 --out atomic-declared.json
run p1-float $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --out p1-float.json
T64P2=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var/p2.exact_f64.chunk512.as_f32
if [ -f $T64P2/meta.json ]; then
  run p2-float-vs-f64 $BIN/ink2-gpu $M --class float --tape $T64P2 --score --out p2-float-vs-f64.json
  run p2-declared-vs-f64 $BIN/ink2-gpu $M --class declared --tape $T64P2 --score --out p2-declared-vs-f64.json
else
  echo "SKIP p2 vs f64: no tape yet"
fi
